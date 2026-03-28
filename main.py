"""
量化训练主入口 - 整合所有量化功能
提供完整的量化训练流程
"""
import argparse
import os
import math
import random
import torch
import torch.nn as nn
from typing import Dict, Optional
import json
from pathlib import Path
from timm.scheduler import create_scheduler_v2
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from model import RecNetwork
from loss import RecognitionLoss
from data import VOCAB_SIZE, idx2char, char2idx, blank_id, sos_id, eos_id, VOCAB, OTHER_PAD_SIZE, CONFUSE_WEIGHT_OPTIMIZED, create_dataset, create_dataset_splitted
from debug import debug_virtual_alignment
from deployment import create_deployment_package, ctc_decode_v2, cer_score, exact_match
from quantization import (
    QUANTIZATION_CONFIG,
    QuantizationStrategy, QuantizationManager,
    QuantizationConfig, AdaptiveQuantizationConfig,
    QuantizationEvaluator, QuantizationMetrics,
    create_optimal_config, get_config_template, create_optimization_study,
    PruningManager, create_pruning_config,
    QATTrainingScheduler,  # 优化3: 分阶段QAT训练调度器
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='OCR量化感知训练工具')

    # 基础配置
    parser.add_argument('--config', default='quantization_config.json', type=str, help='量化配置文件路径')
    parser.add_argument('--template', type=str, choices=[
        'ocr_conservative', 'ocr_balanced', 'ocr_aggressive',
        'mobile_optimized', 'server_optimized'
    ], default='ocr_balanced', help='使用预定义配置模板')

    # 训练配置
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'deployment', 'optimization_study', 'both'],
                       default='both', help='运行模式')
    parser.add_argument('--model_name', type=str, default='viptr2', help='训练模型名称')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--train_mode', type=str, choices=['ctc', 'ar', 'hybrid'],
                       default='ctc', help='训练模式: ctc 仅训练CTC解码头 ar 仅训练自回归解码器 hybrid 同时训练两个解码头')
    parser.add_argument('--num_layers', type=int, default=3, help='网络层数')
    parser.add_argument('--num_heads', type=int, default=6, help='注意力头数')
    parser.add_argument('--d_model', type=int, default=384, help='模型隐藏维度')
    parser.add_argument('--in_channels', type=int, default=3, help='输入通道数')
    parser.add_argument('--max_text_length', type=int, default=70, help='最大文本长度')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout比例')
    parser.add_argument('--num_workers', type=int, default=0, help='数据集加载进程数')
    parser.add_argument('--warmup_lr', type=int, default=3, help='前 N 轮线性增大学习率')
    parser.add_argument('--warmup_decoder', type=int, default=10, help='前 N 轮不开启蒸馏')

    # 数据配置
    parser.add_argument('--num_train', type=int, default=95000, help='训练样本数')
    parser.add_argument('--num_val', type=int, default=5000, help='验证样本数')
    parser.add_argument('--img_height', type=int, default=32, help='图像高度')
    parser.add_argument('--img_min_width', type=int, default=128, help='图像最小宽度')
    parser.add_argument('--img_export_width', type=int, default=512, help='图像最小宽度')
    parser.add_argument('--min_chars', type=int, default=15, help='最少字符数')
    parser.add_argument('--max_chars', type=int, default=50, help='最多字符数')
    parser.add_argument('--dataset_type', type=str, choices=['synthetic', 'lmdb'], default='synthetic', help='数据集类型')
    parser.add_argument('--lmdb_data_dir', type=str, default=None, help='LMDB数据集目录路径')
    parser.add_argument('--key_img_prefix', type=str, default='img-', help='LMDB图像key前缀')
    parser.add_argument('--key_label_prefix', type=str, default='label-', help='LMDB标签key前缀')

    # 模型保存配置
    parser.add_argument('--checkpoint', type=str, default='ocr_latest.pth', help='断点文件路径')
    parser.add_argument('--checkpoint_cer', type=str, default='ocr_best_cer.pth', help='最佳CER模型文件路径')
    parser.add_argument('--checkpoint_em', type=str, default='ocr_best_em.pth', help='最佳EM模型文件路径')

    # 量化配置
    parser.add_argument('--enable_quantization', action='store_true', default=False,
                       help='启用量化训练')
    parser.add_argument('--quantization_mode', type=str, choices=['qat', 'ptq', 'both'], default='qat',
                       help='量化模式: qat 量化感知训练, ptq 训练后量化, both 两者都启用')
    parser.add_argument('--quantization_strategy', type=str,
                       choices=['int8_dyn_act_int4_weight', 'int8_weight_only',
                               'int4_weight_only', 'int8_dynamic_activation_int8_weight'],
                       default='int8_dyn_act_int4_weight', help='量化策略')
    parser.add_argument('--qat_epochs', type=int, default=5, help='QAT训练轮数')
    parser.add_argument('--weight_bits', type=int, default=4, help='权重量化位数')
    parser.add_argument('--activation_bits', type=int, default=8, help='激活量化位数')

    # 优化1: 分层混合精度QAT
    parser.add_argument('--enable_layer_wise_qat', action='store_true', default=False,
                       help='启用分层混合精度QAT (ComposableQATQuantizer)')

    # 优化3: 分阶段QAT训练
    parser.add_argument('--qat_insert_epoch', type=int, default=None,
                       help='QAT FakeQuantize插入的epoch (默认与warmup_lr相同)')

    # 量化优化研究配置
    parser.add_argument('--study_name', type=str, help='优化研究名称')
    parser.add_argument('--method', type=str, default='bayesian',
                       choices=['bayesian', 'grid_search', 'random_search'],
                       help='优化方法')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='优化试验次数')
    parser.add_argument('--param_config', type=str, help='参数配置文件路径')
    parser.add_argument('--optimization_target', type=str, default='balanced',
                       choices=['balanced', 'accuracy', 'compression', 'speed'],
                       help='优化目标')
    parser.add_argument('--dry_run', action='store_true', default=False,
                       help='是否只验证代码流程，不执行实际优化')

    # 硬件配置
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'], help='计算设备')
    parser.add_argument('--hardware_target', type=str, default='cpu',
                       choices=['cpu', 'gpu', 'mobile'], help='目标硬件平台')

    # 输出配置
    parser.add_argument('--output_dir', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--save_models', action='store_true', default=True,
                       help='保存模型文件')
    parser.add_argument('--generate_report', action='store_true', default=True,
                       help='生成评估报告')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='生成可视化图表')

    # 剪枝配置
    parser.add_argument('--enable_pruning', action='store_true', default=False,
                       help='启用模型剪枝')
    parser.add_argument('--pruning_strategy', type=str,
                       choices=['l1_unstructured', 'l1_structured', 'l2_structured', 'ln_structured', 'global_unstructured'],
                       default='l1_unstructured', help='剪枝策略')
    parser.add_argument('--pruning_ratio', type=float, default=0.3,
                       help='全局剪枝比例')
    parser.add_argument('--pruning_layers', type=str, nargs='+',
                       default=['backbone', 'neck', 'decoder'], help='需要剪枝的层')
    parser.add_argument('--pruning_epoch', type=int, default=20,
                       help='剪枝执行的epoch')
    parser.add_argument('--min_acc_drop', type=float, default=0.01,
                       help='允许的最大精度下降')
    parser.add_argument('--finetune_epochs', type=int, default=10,
                       help='剪枝后的微调轮数')
    parser.add_argument('--prune_criteria', type=str, choices=['l1', 'l2', 'grad'],
                       default='l1', help='剪枝标准')

    # 分层剪枝比例
    parser.add_argument('--backbone_pruning_ratio', type=float, default=0.2,
                       help='Backbone剪枝比例')
    parser.add_argument('--neck_pruning_ratio', type=float, default=0.3,
                       help='Neck剪枝比例')
    parser.add_argument('--decoder_pruning_ratio', type=float, default=0.1,
                       help='Decoder剪枝比例')
    
    # 高级剪枝配置
    parser.add_argument('--apply_pruning_during_training', action='store_true', default=False,
                       help='训练期间实际应用剪枝(否则仅在部署时应用)')
    parser.add_argument('--validate_pruning', action='store_true', default=False,
                       help='剪枝后验证精度，如下降过大则回滚')
    parser.add_argument('--visualize_pruning', action='store_true', default=False,
                       help='生成剪枝可视化图表')
    parser.add_argument('--structural_compression', action='store_true', default=False,
                       help='部署时执行结构化压缩(真正删除通道)')

    # 高级配置
    parser.add_argument('--deployment_target', type=str, default='server_cpu',
                       choices=['mobile_cpu', 'mobile_gpu', 'edge_tpu', 'server_cpu', 'server_gpu'], help='部署目标')
    parser.add_argument('--auto_optimize', action='store_true', default=True,
                       help='自动优化量化配置')
    parser.add_argument('--target_compression_ratio', type=float, default=0.25,
                       help='目标压缩比')
    parser.add_argument('--preserve_accuracy', action='store_true', default=True,
                       help='优先保持精度')

    return parser.parse_args()

def count_params(model, name):
    '''统计节点参数量'''
    c = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{name:20s}: {c/1e6:6.2f} M')
    return c

@torch.no_grad()
def calculate_metrics(batch: Dict[str, torch.Tensor], logits: Dict[str, torch.Tensor], labels: torch.Tensor, use_ctc: bool):
    '''计算 CER 和 Exact-Match 指标'''
    em_cnt = 0
    running_cer = 0.0
    skip_tokens = [blank_id, sos_id, eos_id]
    tgt_len = batch['text_lengths'].tolist()

    if use_ctc:
        # 简单去重/去 blank 解码
        pred    = logits['ctc_logits'].argmax(dim=-1).cpu().numpy()               # CTC logit  [B,L,V]
        for p,t,l in zip(pred, labels, tgt_len):
            pp = ctc_decode_v2(p, skip_tokens)
            pred_decoded = torch.tensor(pp, dtype=torch.long)
            running_cer += cer_score(pp, t, l, blank_id, sos_id, eos_id, idx2char)
            em_cnt += exact_match(pred_decoded, t, l, blank_id, eos_id)
    else:
        pred   = logits['ar_logits'].argmax(dim=-1)              # AR logit  [B,L,V]
        for p, t, l in zip(pred, labels[:, 1:], tgt_len):
            running_cer += cer_score(p, t, l, blank_id, sos_id, eos_id, idx2char)
            em_cnt += exact_match(p, t, l, blank_id, eos_id)

    return running_cer, em_cnt

@torch.no_grad()
def validate(device: str, model: torch.nn.Module, loader, criterion, epoch: int, train_mode: str, model_name: str, output_dir: str, max_text_length: int, dtype=torch.float16):
    model.eval()
    use_ctc = True if train_mode == 'ctc' or train_mode == 'hybrid' else False
    running_loss, running_cer, running_em, samples = 0.0, 0.0, 0, 0
    pbar = tqdm(loader, desc='Validate')

    for batch in pbar:
        images = batch['images'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        label_lens = batch['label_lens'].to(device, non_blocking=True)
        labels_ce = batch['labels_ce'].to(device, non_blocking=True)

        with autocast(device, dtype=dtype):
            logits = model(images)               # B×T×V
            loss = criterion(logits, labels, label_lens, targets_ce=labels_ce)

        if use_ctc:
            cer, em_cnt = calculate_metrics(batch, logits, batch['labels'], use_ctc)
        else:
            cer, em_cnt = calculate_metrics(batch, logits, labels_ce, use_ctc)

        bs = images.size(0)
        running_cer += cer
        running_em += em_cnt
        samples += bs
        running_loss += loss['total_loss'].item() * bs
        pbar.set_postfix(loss=f'{running_loss/samples:.4f}',
                         cer=f'{running_cer/samples:.4f}',
                         acc=f'{running_em/samples:.2%}')

    debug_virtual_alignment(device, model, loader, epoch, model_name, train_mode, output_dir,
                          blank_id=blank_id, eos_id=eos_id, idx2char=idx2char, sos_id=sos_id,
                          max_length=max_text_length)

    return running_loss/samples, running_cer/samples, running_em/samples

def train_one_epoch(device: str, model: torch.nn.Module, loader, criterion, optimizer, scaler, epoch: int, train_mode: str, eval_rate: float, dtype=torch.float16):
    model.train()
    running_loss, ctc_loss, ar_loss, kl_loss, distill_loss, feature_loss, quantization_loss, running_cer, running_em, samples = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
    pbar = tqdm(loader, desc=f'Train E{epoch}')
    use_ctc = True if train_mode == 'ctc' or train_mode == 'hybrid' else False
    eval_step = set(sorted(random.sample(range(0, len(loader)), k=int(eval_rate * len(loader)))))

    for idx, batch in enumerate(pbar):
        eval_mode = idx in eval_step
        images = batch['images'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        label_lens = batch['label_lens'].to(device, non_blocking=True)
        labels_ce = batch['labels_ce'].to(device, non_blocking=True)

        ar_mask = (labels_ce != blank_id) & (labels_ce != sos_id) & (labels_ce != eos_id)
        with autocast(device, dtype=dtype):
            # 存储输入特征用于量化损失计算
            logits = model(images, labels_ce, epoch=epoch, eval_mode=eval_mode)
            logits['input_features'] = images

            loss = criterion(logits, labels, label_lens, targets_ce=labels_ce, mask=ar_mask)

        scaler.scale(loss['total_loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3)
        optimizer.zero_grad(set_to_none=True)

        if use_ctc:
            cer, em_cnt = calculate_metrics(batch, logits, batch['labels'], use_ctc)
        else:
            cer, em_cnt = calculate_metrics(batch, logits, labels_ce, use_ctc)

        bs = images.size(0)
        running_cer += cer
        running_em += em_cnt
        samples += bs
        running_loss += loss['total_loss'].item() * bs
        ctc_loss += loss.get('ctc_loss', torch.tensor(0.0)).item() * bs
        ar_loss += loss.get('ar_loss', torch.tensor(0.0)).item() * bs
        kl_loss += loss.get('kl_loss', torch.tensor(0.0)).item() * bs
        distill_loss += loss.get('total_distill_loss', torch.tensor(0.0)).item() * bs
        feature_loss += loss.get('feature_loss', torch.tensor(0.0)).item() * bs
        quantization_loss += loss.get('quantization_loss', torch.tensor(0.0)).item() * bs

        pbar.set_postfix(loss=f'{running_loss/samples:.4f}',
                         ctc_loss=f'{ctc_loss/samples:.4f}',
                         ar_loss=f'{ar_loss/samples:.4f}',
                         kl_loss=f'{kl_loss/samples:.4f}',
                         distill_loss=f'{distill_loss/samples:.4f}',
                         feature_loss=f'{feature_loss/samples:.4f}',
                         quantization_loss=f'{quantization_loss/samples:.4f}',
                         cer=f'{running_cer/samples:.4f}',
                         acc=f'{running_em/samples:.2%}',
                         norm=f'{total_norm:.2f}')
    return running_loss/samples, running_cer/samples, running_em/samples

def find_scaler_cfg(device, model, loader, criterion, optimizer):
    '''搜索最佳梯度缩放参数'''
    # 检查显卡是否支持bf16数据类型
    use_bf16 = False
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.is_bf16_supported()

    dtype = torch.bfloat16 if use_bf16 and device == 'cuda' else torch.float16

    candidates = [
        (64,  100),
        (128, 200),
        (256, 400),
        (512, 800),
        (1024, 1600),
    ]
    for init, grow in candidates:
        scaler = torch.amp.GradScaler(init_scale=init,
                                         growth_interval=grow,
                                         backoff_factor=0.5,
                                         growth_factor=2,
                                         enabled=True)
        model.train()
        overflows = 0
        for step, batch in enumerate(loader):
            images = batch['images'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            label_lens = batch['label_lens'].to(device, non_blocking=True)

            with torch.amp.autocast(device, dtype=dtype):
                out = model(images, labels)
                loss = criterion(out, labels, label_lens)['total_loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale_before = scaler.get_scale()
            scaler.update()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scale_after = scaler.get_scale()
            if scale_after < scale_before:
                overflows += 1
            optimizer.zero_grad(set_to_none=True)

            if step >= 200:
                break
        print(f'init={init:3d} growth={grow:3d} | '
              f'overflows={overflows:3d} | '
              f'final_scale={scaler.get_scale()} | '
              f'total_norm={total_norm}')

def get_quantization_scheduler(epoch: int, config: Dict) -> float:
    """获取量化相关的学习率调度因子"""
    if not config['enabled']:
        return 1.0

    # QAT阶段使用较低的学习率
    if epoch < config['qat_epochs']:
        return config['qat_learning_rate_multiplier']
    else:
        return 1.0

def prepare_model(args, device: str, checkpoint) -> RecNetwork:
    """准备并创建OCR模型

    Args:
        args: 命令行参数，包含模型配置信息
        device: 模型运行设备
        checkpoint: 模型检查点路径，如果存在则加载预训练参数

    Returns:
        创建并初始化的 RecNetwork 模型
    """
    model = RecNetwork(
        train_mode=args.train_mode,
        model_name=args.model_name,
        num_classes=VOCAB_SIZE,
        in_channels=args.in_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_text_length=args.max_text_length,
        hidden_dim=args.d_model,
        dropout=args.dropout,
        pad_token=blank_id,
        sos_token=sos_id,
        eos_token=eos_id,
    ).to(device)

    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        # 使用strict=False加载模型权重，忽略剪枝相关的不匹配键
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        ckpt = None

    return model, ckpt

def train_quantized_main(args, quantization_manager: QuantizationManager, quantization_config: Dict, model: nn.Module, ckpt, device: str, output_dir: str, pruning_manager: PruningManager, confuse_weight_dict: Optional[Dict[int, float]], is_qat: bool = False):
    # 检查显卡是否支持bf16数据类型
    use_bf16 = False
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.is_bf16_supported()

    print(f"💡 bf16支持: {'是' if use_bf16 else '否'}")

    # 优化3: 初始化QAT训练调度器 (分阶段QAT)
    qat_scheduler = None
    if quantization_config.get('enabled') and quantization_config.get('quantization_aware_training') and is_qat:
        # 延迟QAT插入：在warmup阶段正常训练，在指定epoch插入QAT
        qat_scheduler = QATTrainingScheduler(quantization_config, quantization_manager, model, delay_insert=True)
        qat_insert_epoch = quantization_config.get('qat_insert_epoch', 3)
        print(f"📅 初始化QAT分阶段训练调度器")
        print(f"   - 预热阶段: Epoch 0-{qat_insert_epoch-1} (正常精度训练)")
        print(f"   - QAT插入: Epoch {qat_insert_epoch}")
        print(f"   - QAT微调: Epoch {qat_insert_epoch}-{qat_insert_epoch + quantization_config.get('qat_epochs', 5) - 1}")

    # 已有 amp，再打开：
    # 使用新的API设置控制TF32行为，避免PyTorch 2.9后的弃用警告
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.conv.fp32_precision = 'tf32'  # 对于CUDA卷积操作
    #     torch.backends.cuda.matmul.fp32_precision = 'tf32'  # 对于CUDA矩阵乘法操作

    # 使用默认配置如果没有提供
    if quantization_config is None:
        quantization_config = QUANTIZATION_CONFIG

    # 1. 创建DataLoader
    train_loader, val_loader = create_dataset_splitted(
        dataset_type=args.dataset_type,
        data_dir=args.lmdb_data_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        img_height=args.img_height,
        min_width=args.img_min_width,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        key_img_prefix=args.key_img_prefix,
        key_label_prefix=args.key_label_prefix,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if device == 'cuda' else False,
    )

    # 基础损失函数
    criterion = RecognitionLoss(
        max_epoch=args.epochs,
        vocab_size=VOCAB_SIZE,
        ignore_index=blank_id,
        ctc_weight=2.0,
        ar_weight=1.0,
        distill_weight=0.2,
        distill_start_epoch=args.warmup_decoder,
        confuse_weight_dict=confuse_weight_dict,
        quantization_manager=quantization_manager
    ).to(device)

    param_groups = [
        # 1. Backbone
        {"params": model.backbone.parameters(), "lr": args.learning_rate * 0.3},
        # 2. Neck
        {"params": model.neck.parameters(), "lr": args.learning_rate * 1.0},
        # 3. Align
        {"params": model.decoder.feature_align.parameters(), "lr": args.learning_rate * 1.2},
        # 4. CTCHead
        {"params": model.decoder.ctc_decoder.parameters(), "lr": args.learning_rate * 2.0},
        # 5. ArHead
        {"params": model.decoder.ar_decoder.parameters(), "lr": args.learning_rate * 2.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched = 'cosine',
        num_epochs = args.epochs,
        warmup_epochs = args.warmup_lr,
        min_lr = 1e-6
    )

    # 搜索最佳梯度缩放参数，当前 0 overflow 已说明 64~1024 区间都安全，再往上测 2048 意义不大
    # find_scaler_cfg(device, model, train_loader, criterion, optimizer)
    scaler = GradScaler(init_scale=1024, growth_interval=100, enabled=True)

    # 设置autocast设备类型，支持bf16则使用bf16-mixed精度
    dtype = torch.bfloat16 if use_bf16 and device == 'cuda' else torch.float16

    total = 0
    total += count_params(model.backbone,   'Backbone')
    total += count_params(model.neck,   'Neck')
    total += count_params(model.decoder.feature_align,   'Align')
    total += count_params(model.decoder.ctc_decoder,   'CTCHead')
    total += count_params(model.decoder.ar_decoder,   'ArHead')
    print('-'*40)
    print(f'Total Trainable: {total/1e6:6.2f} M')

    # 3. 断点续训
    start_epoch = 0
    if ckpt:
        optimizer.load_state_dict(ckpt['opt'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        print(f'🔥 续训 epoch {start_epoch}')

    # 如果需要PTQ，先进行校准
    if quantization_config['enabled'] and quantization_config['post_training_quantization']:
        print("🔧 开始训练后量化校准...")
        quantization_manager.calibrate_model(train_loader)
        print("✅ PTQ校准完成")

    eval_rate = 0.0
    current_lr = optimizer.param_groups[0]['lr']
    best_cer, best_em, best_em_save = 1.0, 0.0, 0.0
    patience_loss, patience_em = 10, 10   # 连续不改善轮数
    best_train_loss = float('inf')
    counter_loss, counter_em = 0, 0

    # 4. 训练
    success = True
    try:
        # model = torch.compile(model, mode='max-autotune')   # 训练步 10-15 % 提速
        dummy_input = torch.randn(1, 3, args.img_height, args.img_min_width).to(device)
        for epoch in range(start_epoch, args.epochs):
            # 注意: LR显示的是上一轮scheduler.step()后的学习率
            # 实际使用的学习率会在计算乘数后更新
            print(f'\n----- Epoch {epoch} -- LR {current_lr:.6f} -- scale {scaler.get_scale()} -----')

            # 调整学习率 - 优化3: 集成QAT训练调度器
            lr_multipliers = []

            # QAT分阶段学习率调整 (新版调度器)
            if qat_scheduler is not None:
                # 检查是否需要插入QAT（延迟插入）
                if not qat_scheduler.qat_inserted:
                    qat_inserted = qat_scheduler.maybe_insert_qat(epoch)
                    if qat_inserted:
                        print(f"🎯 Epoch {epoch}: QAT量化模块已插入，开始量化感知训练")

                stage = qat_scheduler.get_current_stage(epoch)
                stage_lr_multiplier = qat_scheduler.get_lr_multiplier(epoch)
                lr_multipliers.append(stage_lr_multiplier)

                # 检查是否需要冻结BN/LN层（仅在QAT微调阶段且尚未冻结时）
                if qat_scheduler.should_freeze_bn_ln(epoch) and not qat_scheduler._bn_frozen:
                    qat_scheduler.freeze_bn_ln_for_qat(model)
                    print(f"🧊 Epoch {epoch}: 进入QAT微调阶段，冻结归一化层")
            else:
                # 回退到旧版QAT学习率调整
                qat_lr_multiplier = get_quantization_scheduler(epoch, quantization_config)
                lr_multipliers.append(qat_lr_multiplier)

            # 剪枝微调阶段学习率调整
            finetune_lr_multiplier = pruning_manager.get_finetune_lr_multiplier(epoch)
            lr_multipliers.append(finetune_lr_multiplier)

            # 总学习率乘数
            total_lr_multiplier = 1.0
            for multiplier in lr_multipliers:
                total_lr_multiplier *= multiplier

            # 修复: 应用学习率调整
            # 先让scheduler计算当前epoch的基础学习率，再应用乘数
            scheduler_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.learning_rate
            new_lr = scheduler_lr * total_lr_multiplier
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            # 打印当前阶段信息
            if qat_scheduler is not None:
                stage = qat_scheduler.get_current_stage(epoch)
                qat_status = "[已插入]" if qat_scheduler.qat_inserted else "[等待插入]"
                print(f'   [阶段: {stage}] {qat_status} [LR乘数: {total_lr_multiplier:.3f}]')

            train_loss, train_cer, train_em = train_one_epoch(device, model, train_loader,
                                                            criterion, optimizer, scaler, epoch, args.train_mode, eval_rate, dtype=dtype)
            val_loss, val_cer, val_em = validate(device, model, val_loader, criterion, epoch, args.train_mode, args.model_name, output_dir, args.max_text_length, dtype=dtype)

            criterion.schedule(epoch)
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']

            # 训练模式正确率达到 60% 开始让训练走推理分支，解决模型训练时不知道自回归推理错误字符
            if train_em > 0.6:
                eval_rate = 0.3

            # ---------- 1. 训练 loss 异常 ----------
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                success = False
                print('❌ Train | Val loss 异常，立即终止训练')
                break

            # ---------- 2. 监控训练指标不提升就早停 ----------
            # ---------- Train loss 10 轮不下降 ----------
            if train_loss < best_train_loss - 1e-5:
                best_train_loss = train_loss
                counter_loss = 0
            elif epoch > 5:
                counter_loss += 1
                if counter_loss >= patience_loss:
                    print(f'⚠️  Train loss 连续 {patience_loss}  epoch 无下降，提前停止')
                    break

            # ---------- Val_EM 10 轮不涨 ----------
            if val_em > best_em + 1e-5:
                best_em = val_em
                counter_em = 0
            elif epoch > 5:
                counter_em += 1
                if counter_em >= patience_em:
                    print(f'⚠️  Val_EM 连续 {patience_em}  epoch 无提升，提前停止')
                    break

            # ---------- 3. 保存模型 ----------
            quantization_manager.export_quantized_model(pruning_manager.pruning_applied, f'{output_dir}/models/{args.checkpoint}', epoch, best_cer, best_em,
                                                        dummy_input, optimizer.state_dict(), scaler.state_dict(), model)
            if val_cer > 0 and val_cer < best_cer:
                best_cer = val_cer
                quantization_manager.export_quantized_model(pruning_manager.pruning_applied, f'{output_dir}/models/{args.checkpoint_cer}', epoch, best_cer, best_em,
                                                            dummy_input, optimizer.state_dict(), scaler.state_dict(), model)
                print(f'✅ 最佳 CER 模型已保存  CER={best_cer:.2%}')
            if val_em > best_em_save:      # 与上面 best_em 分开，避免提前停止干扰
                best_em_save = val_em
                quantization_manager.export_quantized_model(pruning_manager.pruning_applied, f'{output_dir}/models/{args.checkpoint_em}', epoch, best_cer, best_em,
                                                            dummy_input, optimizer.state_dict(), scaler.state_dict(), model)
                print(f'🚀 最佳 Exact-Match 模型已保存  EM={best_em:.2%}')

            # ---------- 4. Val_EM 达标退出训练 ----------
            if val_em >= 0.992:
                print('🎉  Val_EM 达到 99.2 %，训练提前完成')
                break

            # 检查是否需要执行剪枝
            if pruning_manager.is_pruning_time(epoch):
                print(f"\n{'='*60}")
                print(f"🌳 执行模型剪枝 (Epoch {epoch})")
                print(f"{'='*60}")
                
                # 方案1: 训练期间应用剪枝
                if getattr(args, 'apply_pruning_during_training', False):
                    # 根据策略选择剪枝方法
                    if args.pruning_strategy == 'global_unstructured':
                        pruning_applied = pruning_manager.apply_global_pruning(
                            epoch, val_em, best_em
                        )
                    else:
                        pruning_applied = pruning_manager.apply_pruning(
                            epoch, val_em, best_em
                        )
                    
                    if pruning_applied:
                        print(f"✅ 剪枝已应用")
                        
                        # 验证剪枝效果（如果启用）
                        if getattr(args, 'validate_pruning', False):
                            print("🔍 验证剪枝效果...")
                            is_valid = pruning_manager.validate_pruning_with_rollback(
                                val_loader, 
                                max_acc_drop=args.min_acc_drop
                            )
                            if not is_valid:
                                print("⚠️ 剪枝导致精度下降过大，已自动回滚")
                            else:
                                print("✅ 剪枝验证通过")
                        
                        # 生成可视化（如果启用）
                        if getattr(args, 'visualize_pruning', False):
                            vis_path = f'{output_dir}/visualizations/pruning_epoch_{epoch}.png'
                            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                            pruning_manager.visualize_pruning(vis_path)
                        
                        # 打印剪枝统计
                        ratio = pruning_manager.calculate_pruning_ratio()
                        struct_ratio, pruned_ch, total_ch = pruning_manager.calculate_structured_pruning_ratio()
                        print(f"📊 剪枝统计:")
                        print(f"   - 参数剪枝比例: {ratio:.2%}")
                        print(f"   - 结构化剪枝: {pruned_ch}/{total_ch} channels ({struct_ratio:.2%})")
                    else:
                        print("⚠️ 剪枝未应用（可能因精度不达标）")
                
                # 方案2: 仅记录候选节点（部署时应用）
                else:
                    pruning_manager.record_pruning_candidates(epoch, val_em, best_em)
                    print("✅ 剪枝候选节点已记录（将在部署时应用）")
                
                print(f"{'='*60}\n")

    except Exception as e:
        print(f'❌ 训练过程中出现异常: {e}')
        import traceback
        traceback.print_exc()
        success = False

    print('\n训练结束（可能提前停止）')

    # 保存剪枝候选信息（如果有）
    if pruning_manager.pruning_candidates:
        candidates_path = f'{output_dir}/models/pruning_candidates.json'
        pruning_manager.save_pruning_candidates(candidates_path)
        
    # 保存剪枝可视化（如果有且未在训练期间生成）
    if (getattr(args, 'visualize_pruning', False) and 
        pruning_manager.pruning_applied and 
        not getattr(args, 'apply_pruning_during_training', False)):
        vis_path = f'{output_dir}/visualizations/pruning_final.png'
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        pruning_manager.visualize_pruning(vis_path)

    return model, success

def setup_device(device: str) -> str:
    """设置计算设备"""
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def create_quantization_config(args, model: nn.Module, config_file: str) -> QuantizationConfig:
    """创建量化配置"""

    # 如果有配置文件就加载
    if os.path.exists(config_file):
        print(f"📄 从文件加载量化配置: {config_file}")
        config = QuantizationConfig.load_from_file(config_file)
    else:
        # 使用预定义模板或自动优化
        if args.auto_optimize:
            print("🎯 使用自动优化配置")
            config = create_optimal_config(
                model=model,
                task_type='ocr',
                hardware_target=args.hardware_target,
                target_compression_ratio=args.target_compression_ratio,
                preserve_accuracy=args.preserve_accuracy
            )
        else:
            print(f"📋 使用预定义模板: {args.template}")
            config = get_config_template(args.template)

        # 覆盖命令行参数
        config.enabled = args.enable_quantization
        config.strategy = QuantizationStrategy(args.quantization_strategy)
        config.qat_epochs = args.qat_epochs
        config.weight_bits = args.weight_bits
        config.activation_bits = args.activation_bits

    return config

def prepare_output_directory(args) -> str:
    """准备输出目录"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)

    return str(output_dir)

def prepare_quantized_manage(args, config: QuantizationConfig, model: nn.Module, output_dir: str):
    """初始化量化管理器"""
    print("🚀 开始准备量化感知训练参数...")

    # 设置训练参数
    train_config = {
        'IMG_H': args.img_height,
        'IMG_MIN_W': args.img_min_width,
        'MIN_CHARS': args.min_chars,
        'MAX_CHARS': args.max_chars,
        'BATCH_SIZE': args.batch_size,
        'NUM_EPOCHS': args.epochs,
        'LR': args.learning_rate,
        'NUM_TRAIN': args.num_train,
        'NUM_VAL': args.num_val,
        'NUM_LAYERS': args.num_layers,
        'NUM_HEADS': args.num_heads,
        'D_MODEL': args.d_model,
        'IN_CHANNELS': args.in_channels,
        'MAX_TEXT_LENGTH': args.max_text_length,
        'DROPOUT': args.dropout,
        'NUM_WORKERS': args.num_workers,
        'TRAIN_MODE': args.train_mode,
        'MODEL_NAME': args.model_name,
        'WARMUP_LR': args.warmup_lr,
        'WARMUP_DECODER': args.warmup_decoder,
        'CHECKPOINT': os.path.join(output_dir, 'models', 'ocr_latest.pth'),
        'CHECKPOINT_CER': os.path.join(output_dir, 'models', 'ocr_best_cer.pth'),
        'CHECKPOINT_EM': os.path.join(output_dir, 'models', 'ocr_best_em.pth'),
        'EXPORT_PATH': os.path.join(output_dir, 'models', 'quantized_ocr_model.pth'),
        'OUTPUT_DIR': output_dir,
    }

    # 保存训练配置
    if args.mode in ['train', 'both']:
        config_path = Path(output_dir) / 'train_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(train_config, f, indent=2, ensure_ascii=False)

    # 运行量化训练
    print("📊 训练配置:")
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    print("📊 训练参数:")
    print(json.dumps(train_config, indent=2, ensure_ascii=False))

    # 将QuantizationConfig转换为字典
    quantization_config = config.to_dict()
    # 根据命令行参数设置量化模式
    if args.quantization_mode == 'qat':
        quantization_config['quantization_aware_training'] = True
        quantization_config['post_training_quantization'] = False
    elif args.quantization_mode == 'ptq':
        quantization_config['quantization_aware_training'] = False
        quantization_config['post_training_quantization'] = True
    else:  # both
        quantization_config['quantization_aware_training'] = True
        quantization_config['post_training_quantization'] = True
    quantization_config['quantization_strategy'] = args.quantization_strategy
    quantization_config['weight_bits'] = args.weight_bits
    quantization_config['activation_bits'] = args.activation_bits
    quantization_config['quantization_granularity'] = 'per_channel'

    # 优化1&2: 传递新配置参数
    quantization_config['enable_layer_wise_qat'] = args.enable_layer_wise_qat
    quantization_config['qat_insert_epoch'] = args.qat_insert_epoch if args.qat_insert_epoch is not None else args.warmup_lr

    if quantization_config['enable_layer_wise_qat']:
        print(f"🔧 分层混合精度QAT: 启用")
    quantization_config['observer_type'] = 'moving_average'
    quantization_config['observer_momentum'] = 0.1

    # 保存配置
    if args.mode in ['train', 'both']:
        config_path = Path(output_dir) / 'quantization_config.json'
        config.save_to_file(str(config_path))
        print(f"💾 量化配置已保存: {config_path}")

    print("✅ 初始化量化管理器...")
    quantization_manager = QuantizationManager(model, quantization_config)

    # 判断是否需要延迟QAT插入
    # QAT模式: 延迟插入，在指定epoch再应用QAT
    # PTQ模式: 立即应用（无需训练阶段插入）
    is_qat = quantization_config.get('quantization_aware_training', False)
    is_ptq = quantization_config.get('post_training_quantization', False)

    if is_qat:
        # QAT模式: 延迟到指定epoch插入
        qat_insert_epoch = quantization_config.get('qat_insert_epoch', 3)
        print(f"🎯 QAT模式: 将在 Epoch {qat_insert_epoch} 插入量化模块")
        print("   预热阶段使用正常精度训练...")
        # 保存原始模型用于知识蒸馏（通常在prepare_model_for_quantization中完成，但这里延迟调用）
        import copy
        quantization_manager.original_model = copy.deepcopy(model)
        quantization_manager.original_model.eval()
        # 不立即调用prepare_model_for_quantization()
    else:
        # PTQ模式或量化禁用: 立即应用（或跳过）
        model = quantization_manager.prepare_model_for_quantization()

    return quantization_manager, quantization_config, model, is_qat

def evaluate_quantization(original_model: nn.Module, quantized_model: nn.Module,
                         args, config: QuantizationConfig, device: str, output_dir: str,
                         quantization_manager: 'QuantizationManager' = None):
    """评估量化效果

    Args:
        quantization_manager: 量化管理器，用于转换新版API的FakeQuantize模型
    """
    print("🔍 开始量化效果评估...")

    # 如果是新版API的FakeQuantize模型，先转换为真实量化模型
    if quantization_manager is not None:
        if hasattr(quantization_manager, 'use_modern_api') and quantization_manager.use_modern_api:
            if hasattr(quantization_manager, 'convert_to_quantized_model'):
                print("🔄 评估前转换FakeQuantize模型为真实量化模型...")
                try:
                    quantized_model = quantization_manager.convert_to_quantized_model(quantized_model)
                    print("✅ 模型转换完成，使用真实量化模型进行评估")
                except Exception as e:
                    print(f"⚠️  模型转换失败，使用原始量化模型进行评估: {e}")

    # 创建评估器
    evaluator = QuantizationEvaluator(original_model, quantized_model, device)

    # 创建数据集 - 使用更真实的配置
    print("📊 准备评估数据...")
    print(f"   - 样本数量: {args.num_val}")
    print(f"   - 图像尺寸: {args.img_height}x{args.img_min_width}")
    print(f"   - 字符范围: {args.min_chars}-{args.max_chars}")

    # 创建评估DataLoader
    val_loader = create_dataset(
        dataset_type=args.dataset_type,
        data_dir=args.lmdb_data_dir,
        num_samples=args.num_val,
        img_height=args.img_height,
        min_width=args.img_min_width,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        key_img_prefix=args.key_img_prefix,
        key_label_prefix=args.key_label_prefix,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if device == 'cuda' else False,
    )

    # 运行评估
    print("📊 运行全面评估...")
    print(f"   - 评估批次: {min(50, len(val_loader))}")
    print(f"   - 评估设备: {device}")

    metrics = evaluator.evaluate_quantization(val_loader, num_batches=min(50, len(val_loader)))

    # 生成报告
    if args.generate_report:
        report_path = Path(output_dir) / 'reports' / 'quantization_report.json'
        report = evaluator.generate_report(metrics, str(report_path))
        print(f"📄 评估报告已生成: {report_path}")

    # 生成可视化
    if args.visualize:
        viz_path = Path(output_dir) / 'visualizations' / 'quantization_results.png'
        evaluator.visualize_results(metrics, str(viz_path))
        print(f"📈 可视化图表已生成: {viz_path}")

    return metrics

def save_models(original_model: nn.Module, quantized_model: nn.Module,
                args, config: QuantizationConfig, output_dir: str):
    """保存模型文件"""
    if not args.save_models:
        return

    models_dir = Path(output_dir) / 'models'

    # 保存原始模型
    original_path = models_dir / 'original_model.pth'
    torch.save({
        'model_state_dict': original_model.state_dict(),
        'config': config.to_dict(),
        'model_type': 'original'
    }, original_path)
    print(f"💾 原始模型已保存: {original_path}")

    # 保存量化模型
    quantized_path = models_dir / 'quantized_model.pth'
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'config': config.to_dict(),
        'model_type': 'quantized'
    }, quantized_path)
    print(f"💾 量化模型已保存: {quantized_path}")

    # 导出量化模型（使用torch.export）
    try:
        dummy_input = torch.randn(1, 3, args.img_height, args.img_export_width).to(next(quantized_model.parameters()).device)

        # 使用torch.export导出量化模型
        from torch.export import export

        # 确保模型在评估模式
        quantized_model.eval()

        # 导出模型
        exported_program = export(quantized_model, (dummy_input,))

        # 保存导出的程序
        exported_path = models_dir / 'quantized_model_exported.pt2'
        torch.save(exported_program, exported_path)
        print(f"💾 量化模型已导出 (torch.export): {exported_path}")

    except Exception as e:
        print(f"⚠️  torch.export导出失败: {e}")

    # 保存ONNX格式（如果可能）
    try:
        dummy_input = torch.randn(1, 3, args.img_height, args.img_export_width)

        # 原始模型ONNX
        onnx_original_path = models_dir / 'original_model.onnx'
        torch.onnx.export(
            original_model, dummy_input, str(onnx_original_path),
            input_names=['x'], output_names=['output'],
            dynamic_axes={'x': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"💾 原始模型ONNX已保存: {onnx_original_path}")

        # 量化模型ONNX
        onnx_quantized_path = models_dir / 'quantized_model.onnx'
        torch.onnx.export(
            quantized_model, dummy_input, str(onnx_quantized_path),
            input_names=['x'], output_names=['output'],
            dynamic_axes={'x': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"💾 量化模型ONNX已保存: {onnx_quantized_path}")

    except Exception as e:
        print(f"⚠️  ONNX导出失败: {e}")

def print_summary(metrics: QuantizationMetrics, config: QuantizationConfig, output_dir: str):
    """打印总结信息"""
    print("\n" + "="*60)
    print("🎯 量化训练总结")
    print("="*60)

    print("📊 精度影响:")
    print(f"  - 原始准确率: {metrics.original_accuracy:.4f}")
    print(f"  - 量化准确率: {metrics.quantized_accuracy:.4f}")
    print(f"  - 精度下降: {metrics.accuracy_drop:.4f} ({metrics.accuracy_drop_ratio*100:.2f}%)")

    print("\n📏 模型压缩:")
    print(f"  - 原始大小: {metrics.original_model_size_mb:.1f} MB")
    print(f"  - 量化大小: {metrics.quantized_model_size_mb:.1f} MB")
    print(f"  - 压缩比: {metrics.compression_ratio:.2f}x")
    print(f"  - 大小减少: {metrics.size_reduction_ratio*100:.1f}%")

    print("\n⚡ 推理加速:")
    print(f"  - 原始推理时间: {metrics.original_inference_time_ms:.2f} ms")
    print(f"  - 量化推理时间: {metrics.quantized_inference_time_ms:.2f} ms")
    print(f"  - 加速比: {metrics.speedup_ratio:.2f}x")

    print("\n💾 内存优化:")
    print(f"  - 内存减少: {metrics.memory_reduction_ratio*100:.1f}%")

    print("\n🔧 量化配置:")
    print(f"  - 策略: {config.strategy}")
    print(f"  - 权重量化: {config.weight_bits}位")
    print(f"  - 激活量化: {config.activation_bits}位")
    print(f"  - QAT轮数: {config.qat_epochs}")

    print(f"\n📁 输出目录: {output_dir}")
    print("="*60)

def main():
    """主函数"""
    args = parse_args()

    # 设置设备
    success = True
    device = setup_device(args.device)
    print(f"🔧 使用设备: {device}")

    # 准备输出目录
    output_dir = prepare_output_directory(args)
    print(f"📁 输出目录: {output_dir}")

    # 创建模型
    print("🏗️  创建OCR模型...")
    checkpoint = f'{output_dir}/models/{args.checkpoint}'
    model, ckpt = prepare_model(args, device, checkpoint)

    # 创建量化配置
    config_file = Path(output_dir) / args.config
    config = create_quantization_config(args, model, str(config_file))
    print("⚙️  量化配置创建完成")

    # 初始化量化管理器
    quantization_manager, quantization_config, model, is_qat = prepare_quantized_manage(args, config, model, output_dir)

    # 创建剪枝配置和管理器
    pruning_config = create_pruning_config(args)
    pruning_manager = PruningManager(pruning_config, model)
    print("🌱 剪枝配置创建完成")

    # 如果进程数量为零且是CUDA训练，将进程数量设置为CPU核心数
    if args.num_workers == 0 and device == 'cuda':
        args.num_workers = os.cpu_count()
        print(f"⚠️  未指定进程数，自动设置为CPU核心数: {args.num_workers}")

    # 训练模式
    if args.mode in ['train', 'both']:
        # 调用主要的训练函数
        print("🎯 开始量化训练...")
        # 将字符为键的混淆权重字典转换为索引为键的字典
        confuse_weight_dict_idx = {char2idx[char]: weight for char, weight in CONFUSE_WEIGHT_OPTIMIZED.items() if char in char2idx}
        quantized_model, success = train_quantized_main(
            args=args,
            quantization_manager=quantization_manager,
            quantization_config=quantization_config,
            model=model,
            ckpt=ckpt,
            device=device,
            output_dir=output_dir,
            confuse_weight_dict=confuse_weight_dict_idx,
            pruning_manager=pruning_manager,
            is_qat=is_qat
        )

        if quantized_model is None:
            # 如果训练函数返回None，使用原始模型作为量化模型（降级处理）
            print("⚠️  量化训练失败，使用原始模型进行评估")
            quantized_model = model
        else:
            print("✅ 成功获得量化模型")
    else:
        # 评估模式，需要先加载或创建量化模型
        print("⚠️  评估模式需要先训练或加载量化模型")
        quantized_model = model

    # 评估模式
    if success and args.mode in ['evaluate', 'both']:
        # 评估量化效果
        metrics = evaluate_quantization(model, quantized_model, args, config, device, output_dir, quantization_manager)

        # 保存模型
        save_models(model, quantized_model, args, config, output_dir)

        # 打印总结
        print_summary(metrics, config, output_dir)

    # 创建部署
    if success and args.mode in ['deployment', 'both']:
        if args.enable_pruning:
            print(f"\n{'='*60}")
            print("🌳 部署阶段剪枝处理")
            print(f"{'='*60}")
            
            # 检查是否已经应用了剪枝（训练期间应用）
            if not pruning_manager.pruning_applied:
                # 检查是否有剪枝候选信息文件
                candidates_path = f'{output_dir}/models/pruning_candidates.json'
                if os.path.exists(candidates_path):
                    print(f"📄 加载剪枝候选信息: {candidates_path}")
                    pruning_manager.load_pruning_candidates(candidates_path)
                    
                    # 应用剪枝
                    pruning_manager.apply_pruning_from_candidates()
                else:
                    print(f"⚠️  未找到剪枝候选信息文件，尝试直接应用剪枝...")
                    # 直接应用剪枝（如果配置了剪枝比例）
                    if args.pruning_strategy == 'global_unstructured':
                        pruning_manager.apply_global_pruning(
                            args.pruning_epoch, current_acc=1.0, best_acc=1.0
                        )
                    else:
                        pruning_manager.apply_pruning(
                            args.pruning_epoch, current_acc=1.0, best_acc=1.0
                        )
            
            # 使剪枝永久化
            if pruning_manager.pruning_applied:
                print("🔧 使剪枝永久化...")
                pruning_manager.make_pruning_permanent()
                
                # 结构化压缩（如果启用）
                if getattr(args, 'structural_compression', False):
                    print("🔧 执行结构化压缩...")
                    print("   注意: 这将真正删除被剪枝的通道，减少模型大小")
                    
                    # 创建验证加载器用于验证压缩效果
                    val_loader = create_dataset(
                        dataset_type=args.dataset_type,
                        data_dir=args.lmdb_data_dir,
                        num_samples=min(1000, args.num_val),
                        img_height=args.img_height,
                        min_width=args.img_min_width,
                        min_chars=args.min_chars,
                        max_chars=args.max_chars,
                        key_img_prefix=args.key_img_prefix,
                        key_label_prefix=args.key_label_prefix,
                    )
                    
                    pruning_manager.compress_model_structurally(
                        val_dataloader=val_loader,
                        max_acc_drop=args.min_acc_drop
                    )
                
                # 打印剪枝信息
                prune_info = pruning_manager.get_pruned_model_info()
                stats = pruning_manager.get_detailed_pruning_stats()
                struct_ratio, pruned_ch, total_ch = pruning_manager.calculate_structured_pruning_ratio()
                
                print(f"\n📊 剪枝总结:")
                print(f"   - 剪枝应用: {'是' if prune_info['pruning_applied'] else '否'}")
                print(f"   - 剪枝层数量: {prune_info['pruned_layers_count']}")
                print(f"   - 参数剪枝比例: {stats['global_ratio']:.2%}")
                print(f"   - 结构化剪枝: {pruned_ch}/{total_ch} channels ({struct_ratio:.2%})")
                print(f"   - 总参数量: {stats['total_params']:,}")
                print(f"   - 剪枝参数量: {stats['pruned_params']:,}")
                
                # 生成可视化
                if getattr(args, 'visualize_pruning', False):
                    vis_path = f'{output_dir}/visualizations/pruning_deployment.png'
                    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                    pruning_manager.visualize_pruning(vis_path)
                    print(f"   - 可视化图表: {vis_path}")
            else:
                print("⚠️  剪枝未应用")
            
            print(f"{'='*60}\n")

        dummy_input = torch.randn(3, 3, args.img_height, args.img_export_width)
        create_deployment_package(quantized_model, quantization_config, args.deployment_target, output_dir, dummy_input,
                                vocab=VOCAB, other_pad_size=OTHER_PAD_SIZE,
                                blank_id=blank_id, sos_id=sos_id, eos_id=eos_id,
                                idx2char=idx2char)

    # 创建优化研究
    if args.mode == 'optimization_study':
        # 创建DataLoader
        print("📊 准备优化数据集...")
        val_loader = create_dataset(
            dataset_type=args.dataset_type,
            data_dir=args.lmdb_data_dir,
            num_samples=args.num_val,
            img_height=args.img_height,
            min_width=args.img_min_width,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            key_img_prefix=args.key_img_prefix,
            key_label_prefix=args.key_label_prefix,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False,
            persistent_workers=True if device == 'cuda' else False,
        )

        # 运行优化研究
        create_optimization_study(
            device=device,
            model=model,
            dataloader=val_loader,
            output_dir=output_dir,
            study_name=args.study_name,
            method=args.method,
            n_trials=args.n_trials,
            param_config=args.param_config,
            optimization_target=args.optimization_target,
            dry_run=args.dry_run
        )

    print("✅ 量化训练流程完成！")

if __name__ == "__main__":
    main()
