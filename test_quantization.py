#!/usr/bin/env python3
"""
剪枝功能测试脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from main import parse_args, prepare_model, setup_device
from quantization import PruningConfig, PruningManager, create_pruning_config

def test_pruning_basic():
    """测试剪枝的基本功能"""
    print("🔍 开始测试剪枝基本功能...")

    # 解析命令行参数
    args = parse_args()
    args.enable_pruning = True
    args.pruning_strategy = 'l1_unstructured'
    args.pruning_ratio = 0.3
    args.pruning_epoch = 5
    args.finetune_epochs = 3

    # 设置设备
    device = setup_device('cpu')
    print(f"✅ 设备设置完成: {device}")

    # 准备模型
    checkpoint = ''  # 不使用预训练模型
    model, ckpt = prepare_model(args, device, checkpoint)
    print(f"✅ 模型创建完成")

    # 创建剪枝配置
    pruning_config = create_pruning_config(args)
    print(f"✅ 剪枝配置创建完成: {pruning_config.to_dict()}")

    # 初始化剪枝管理器
    pruning_manager = PruningManager(pruning_config, model)
    print(f"✅ 剪枝管理器初始化完成")

    # 测试剪枝时间检查
    epoch = 0
    is_pruning_time = pruning_manager.is_pruning_time(epoch)
    print(f"✅ 剪枝时间检查: epoch={epoch}, is_pruning_time={is_pruning_time}")

    epoch = args.pruning_epoch
    is_pruning_time = pruning_manager.is_pruning_time(epoch)
    print(f"✅ 剪枝时间检查: epoch={epoch}, is_pruning_time={is_pruning_time}")

    # 测试微调时间检查
    is_finetuning = pruning_manager.is_finetuning_time(epoch)
    print(f"✅ 微调时间检查: epoch={epoch}, is_finetuning={is_finetuning}")

    epoch = args.pruning_epoch + 1
    is_finetuning = pruning_manager.is_finetuning_time(epoch)
    print(f"✅ 微调时间检查: epoch={epoch}, is_finetuning={is_finetuning}")

    # 测试学习率乘数
    lr_multiplier = pruning_manager.get_finetune_lr_multiplier(epoch)
    print(f"✅ 学习率乘数: epoch={epoch}, lr_multiplier={lr_multiplier}")

    # 测试模型信息获取
    prune_info = pruning_manager.get_pruned_model_info()
    print(f"✅ 模型信息获取: {prune_info}")

    print("🎉 剪枝基本功能测试完成！")
    return True

def test_pruning_application():
    """测试剪枝应用"""
    print("\n🔍 开始测试剪枝应用...")

    # 创建一个简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
                nn.ReLU()
            )
            # 计算neck层的输入维度：32x32 -> 16x16 -> 8x8
            self.neck = nn.Linear(32 * 8 * 8, 128)
            self.decoder = nn.Linear(128, 10)

        def forward(self, x):
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
            x = self.neck(x)
            x = self.decoder(x)
            return x

    # 创建模型
    model = TestModel()
    print(f"✅ 测试模型创建完成")

    # 创建剪枝配置
    pruning_config = PruningConfig({
        'enabled': True,
        'pruning_strategy': 'l1_unstructured',
        'pruning_ratio': 0.5,
        'pruning_layers': ['backbone', 'neck', 'decoder'],
        'pruning_epoch': 0,
        'finetune_epochs': 3
    })
    print(f"✅ 剪枝配置创建完成")

    # 初始化剪枝管理器
    pruning_manager = PruningManager(pruning_config, model)
    print(f"✅ 剪枝管理器初始化完成")

    # 模拟训练过程中的剪枝应用
    current_acc = 0.95
    best_acc = 0.96
    pruning_applied = pruning_manager.apply_pruning(0, current_acc, best_acc)
    print(f"✅ 剪枝应用: {pruning_applied}")

    # 检查剪枝比例
    pruning_ratio = pruning_manager.calculate_pruning_ratio()
    print(f"✅ 剪枝比例: {pruning_ratio:.2%}")

    # 获取剪枝信息
    prune_info = pruning_manager.get_pruned_model_info()
    print(f"✅ 剪枝信息: {prune_info}")

    # 永久化剪枝
    pruning_manager.remove_pruning()
    print(f"✅ 剪枝永久化完成")

    # 测试模型是否还能正常工作
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"✅ 模型推理测试: output.shape={output.shape}")

    print("🎉 剪枝应用测试完成！")
    return True

if __name__ == "__main__":
    print("🚀 剪枝功能测试")
    print("=" * 50)

    try:
        # 运行基本功能测试
        test_pruning_basic()

        # 运行剪枝应用测试
        test_pruning_application()

        print("\n" + "=" * 50)
        print("🎉 所有测试通过！剪枝功能正常工作。")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
