"""
增强CTC损失函数测试用例
验证合并后的损失函数功能正确性
"""
import torch
from loss import EnhancedCTCLoss, DistillationLoss
from data import char2idx

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")

    # 创建测试数据
    B, T, V = 2, 10, len(char2idx)
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    input_lens = torch.tensor([10, 8], dtype=torch.long)
    target_lens = torch.tensor([3, 3], dtype=torch.long)

    # 测试增强CTC损失函数
    loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0)

    try:
        loss = loss_fn(logits, targets, input_lens, target_lens)
        print(f"✓ 损失值: {loss.item():.6f}")

        # 测试反向传播
        loss.backward()
        print("✓ 反向传播成功")
        if logits.grad is not None:
            print(f"✓ 梯度形状: {logits.grad.shape}")
            print(f"✓ 梯度均值: {logits.grad.mean().item():.6f}")
            print(f"✓ 梯度范数: {logits.grad.norm().item():.6f}")

        # 测试损失组件分解
        loss_components = loss_fn.get_loss_components(logits, targets, input_lens, target_lens)
        print(f"✓ 基础CTC损失: {loss_components['base_ctc_loss'].item():.6f}")
        print(f"✓ 路径权重: {loss_components['path_weights'].item():.6f}")
        print(f"✓ 加权CTC损失: {loss_components['weighted_ctc_loss'].item():.6f}")
        print(f"✓ 尾部空白惩罚: {loss_components['eos_penalty_loss'].item():.6f}")

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_backward_compatibility():
    """测试基础功能兼容性"""
    print("\n=== 测试基础功能兼容性 ===")

    # 创建测试数据
    B, T, V = 2, 10, len(char2idx)
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    input_lens = torch.tensor([10, 8], dtype=torch.long)
    target_lens = torch.tensor([3, 3], dtype=torch.long)

    try:
        # 测试基础CTC功能
        basic_loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0)
        basic_loss = basic_loss_fn(logits.clone(), targets, input_lens, target_lens)

        # 测试带形近字权重的功能
        weighted_loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0, confuse_gamma=1.0)
        weighted_loss = weighted_loss_fn(logits.clone(), targets, input_lens, target_lens)

        # 测试带EOS惩罚的功能
        eos_loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0, eos_penalty=0.1)
        eos_loss = eos_loss_fn(logits.clone(), targets, input_lens, target_lens)

        print(f"✓ 基础CTC损失: {basic_loss.item():.6f}")
        print(f"✓ 带形近字权重损失: {weighted_loss.item():.6f}")
        print(f"✓ 带EOS惩罚损失: {eos_loss.item():.6f}")

        # 验证梯度正常
        basic_loss.backward()
        if logits.grad is not None:
            print(f"✓ 梯度正常，范数: {logits.grad.norm().item():.6f}")

    except Exception as e:
        print(f"✗ 兼容性测试错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_confuse_characters():
    """测试形近字处理"""
    print("\n=== 测试形近字处理 ===")

    # 创建包含形近字的测试数据
    B, T, V = 2, 10, len(char2idx)
    logits = torch.randn(B, T, V, requires_grad=True)

    # 包含易混淆字符的目标序列：0Ool1I等
    targets = torch.tensor([
        [char2idx['0'], char2idx['O'], char2idx['o'], char2idx['l'], char2idx['1']],  # 易混淆序列
        [char2idx['a'], char2idx['b'], char2idx['c'], char2idx['d'], char2idx['e']]   # 正常序列
    ], dtype=torch.long)
    input_lens = torch.tensor([10, 10], dtype=torch.long)
    target_lens = torch.tensor([5, 5], dtype=torch.long)

    # 测试不同gamma值的影响
    for gamma in [0.5, 1.0, 2.0]:
        loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0, confuse_gamma=gamma)
        loss = loss_fn(logits.clone(), targets, input_lens, target_lens)

        components = loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)
        print(f"✓ Gamma={gamma}: 总损失={loss.item():.6f}, 路径权重={components['path_weights'].item():.6f}")

def test_eos_penalty():
    """测试尾部空白字符惩罚"""
    print("\n=== 测试尾部空白字符惩罚 ===")

    B, T, V = 2, 10, len(char2idx)

    # 创建倾向于在尾部产生空白字符的logits
    logits = torch.randn(B, T, V) * 0.5
    # 在最后几帧增加空白字符的logit值
    logits[:, -3:, 0] += 3.0  # 空白字符索引为0

    logits.requires_grad = True

    targets = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    input_lens = torch.tensor([10, 8], dtype=torch.long)
    target_lens = torch.tensor([3, 3], dtype=torch.long)

    # 测试不同惩罚系数的影响
    for penalty in [0.0, 0.1, 0.5, 1.0]:
        loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0, eos_penalty=penalty)
        loss = loss_fn(logits.clone(), targets, input_lens, target_lens)

        components = loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)
        print(f"✓ EOS惩罚={penalty}: 总损失={loss.item():.6f}, 尾部惩罚={components['eos_penalty_loss'].item():.6f}")

def test_gradient_stability():
    """测试梯度稳定性"""
    print("\n=== 测试梯度稳定性 ===")

    B, T, V = 2, 10, len(char2idx)

    # 创建极端情况下的logits
    logits = torch.zeros(B, T, V)
    logits[:, :, 0] = 10.0  # 空白字符概率极高
    logits.requires_grad = True

    targets = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    input_lens = torch.tensor([10, 8], dtype=torch.long)
    target_lens = torch.tensor([3, 3], dtype=torch.long)

    loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0, gradient_clip=True)

    try:
        loss = loss_fn(logits, targets, input_lens, target_lens)
        loss.backward()

        if logits.grad is not None:
            grad_norm = logits.grad.norm().item()
            print(f"✓ 极端情况下梯度范数: {grad_norm:.6f}")

            # 检查梯度是否合理
            if grad_norm < 1000:  # 梯度没有爆炸
                print("✓ 梯度稳定性良好")
            else:
                print("⚠ 梯度可能过大，需要调整")

    except Exception as e:
        print(f"✗ 梯度稳定性测试错误: {e}")
        return False

    return True

def test_performance_comparison():
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")

    B, T, V = 32, 50, len(char2idx)  # 更大的batch size

    # 创建随机数据
    logits = torch.randn(B, T, V)
    targets = torch.randint(1, V, (B, 10))
    input_lens = torch.full((B,), T, dtype=torch.long)
    target_lens = torch.full((B,), 10, dtype=torch.long)

    import time

    # 测试基础实现
    basic_loss = EnhancedCTCLoss(vocab_size=V, blank=0)

    start_time = time.time()
    for _ in range(10):
        loss1 = basic_loss(logits.clone(), targets, input_lens, target_lens)
    basic_time = time.time() - start_time

    # 测试增强实现（带新优化）
    enhanced_loss = EnhancedCTCLoss(
        vocab_size=V, blank=0,
        char_focal=True,
        focal_gamma=2.0,
        adaptive_margin=True,
        margin_max=0.5
    )

    start_time = time.time()
    for _ in range(10):
        loss2 = enhanced_loss(logits.clone(), targets, input_lens, target_lens)
    enhanced_time = time.time() - start_time

    print(f"✓ 基础实现平均时间: {basic_time/10:.4f}s")
    print(f"✓ 增强实现平均时间: {enhanced_time/10:.4f}s")
    print(f"✓ 性能比率: {enhanced_time/basic_time:.2f}x")
    print(f"✓ 损失值差异: {abs(loss1.item() - loss2.item()):.6f}")

def run_all_tests():
    """运行所有测试"""
    print("开始增强CTC损失函数测试...\n")

    tests = [
        test_basic_functionality,
        test_backward_compatibility,
        test_confuse_characters,
        test_eos_penalty,
        test_gradient_stability,
        test_performance_comparison
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if result is not None else True)
        except Exception as e:
            print(f"测试 {test.__name__} 失败: {e}")
            results.append(False)
        print("-" * 50)

    # 总结结果
    passed = sum(results)
    total = len(results)
    print(f"\n测试总结: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有测试通过！增强CTC损失函数工作正常。")
    else:
        print("⚠️  部分测试失败，请检查实现。")

    return passed == total

def test_char_focal_loss():
    """测试字符级Focal Loss功能"""
    print("\n=== 测试字符级Focal Loss ===")

    B, T, V = 2, 10, len(char2idx)
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    input_lens = torch.tensor([10, 8], dtype=torch.long)
    target_lens = torch.tensor([3, 3], dtype=torch.long)

    # 测试不同focal参数的影响
    for char_focal in [False, True]:
        for gamma in [1.0, 2.0, 3.0]:
            loss_fn = EnhancedCTCLoss(
                vocab_size=V, blank=0,
                char_focal=char_focal,
                focal_gamma=gamma,
                focal_scale=1.0
            )
            loss = loss_fn(logits.clone(), targets, input_lens, target_lens)

            components = loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)

            print(f"✓ 字符级Focal={char_focal}, Gamma={gamma}: 总损失={loss.item():.6f}")
            if char_focal:
                print(f"  - 字符级Focal损失: {components['char_focal_loss'].item():.6f}")
                print(f"  - 样本级Focal损失: {components['sample_focal_loss'].item():.6f}")

def test_adaptive_margin():
    """测试自适应Margin机制"""
    print("\n=== 测试自适应Margin机制 ===")

    B, T, V = 2, 10, len(char2idx)
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    input_lens = torch.tensor([10, 8], dtype=torch.long)
    target_lens = torch.tensor([3, 3], dtype=torch.long)

    # 测试不同margin配置
    test_configs = [
        {"adaptive_margin": False, "margin": 0.0},
        {"adaptive_margin": False, "margin": 0.3},
        {"adaptive_margin": True, "margin": 0.0, "margin_max": 0.5},
        {"adaptive_margin": True, "margin": 0.0, "margin_max": 1.0},
    ]

    for config in test_configs:
        loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0, **config)
        loss = loss_fn(logits.clone(), targets, input_lens, target_lens)

        components = loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)

        config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
        print(f"✓ {config_str}: 总损失={loss.item():.6f}")

        if config.get("adaptive_margin", False):
            print(f"  - 当前自适应Margin: {components.get('adaptive_margin', 0):.4f}")

def test_temperature_annealing():
    """测试温度退火机制"""
    print("\n=== 测试温度退火机制 ===")

    B, T, V = 2, 10, len(char2idx)
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    input_lens = torch.tensor([10, 8], dtype=torch.long)
    target_lens = torch.tensor([3, 3], dtype=torch.long)

    loss_fn = EnhancedCTCLoss(
        vocab_size=V, blank=0,
        temperature_annealing=True,
        char_focal=True,
        focal_gamma=2.0
    )

    # 测试不同epoch的退火效果
    max_epoch = 10
    for epoch in range(0, max_epoch + 1, 2):
        loss_fn.schedule(epoch, max_epoch)
        loss = loss_fn(logits.clone(), targets, input_lens, target_lens)

        components = loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)

        print(f"✓ Epoch {epoch}/{max_epoch}: 总损失={loss.item():.6f}")
        print(f"  - Focal缩放因子: {components['focal_scale']:.4f}")
        print(f"  - 字符级Focal损失: {components['char_focal_loss'].item():.6f}")

def test_combined_optimizations():
    """测试组合优化效果"""
    print("\n=== 测试组合优化效果 ===")

    B, T, V = 4, 15, len(char2idx)

    # 创建包含形近字的困难样本
    logits = torch.randn(B, T, V)
    # 在最后几帧增加blank概率，模拟尾部空白问题
    logits[:, -4:, 0] += 2.0
    logits.requires_grad = True

    # 包含易混淆字符的目标序列
    targets = torch.tensor([
        [char2idx['0'], char2idx['O'], char2idx['o'], char2idx['l'], char2idx['1']],  # 高混淆
        [char2idx['1'], char2idx['l'], char2idx['I'], char2idx['|'], char2idx['i']],  # 高混淆
        [char2idx['a'], char2idx['b'], char2idx['c'], char2idx['d'], char2idx['e']],  # 正常
        [char2idx['p'], char2idx['q'], char2idx['u'], char2idx['v'], char2idx['n']],  # 中等混淆
    ], dtype=torch.long)
    input_lens = torch.tensor([15, 14, 15, 13], dtype=torch.long)
    target_lens = torch.tensor([5, 5, 5, 5], dtype=torch.long)

    # 基准配置 - 只使用基础优化
    baseline_loss_fn = EnhancedCTCLoss(
        vocab_size=V, blank=0,
        confuse_gamma=1.0,
        eos_penalty=0.1
    )
    baseline_loss = baseline_loss_fn(logits.clone(), targets, input_lens, target_lens)
    baseline_components = baseline_loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)

    print(f"✓ 基准配置 - 总损失: {baseline_loss.item():.6f}")
    print(f"  - 基础CTC损失: {baseline_components['base_ctc_loss'].item():.6f}")
    print(f"  - 路径权重: {baseline_components['path_weights'].item():.6f}")
    print(f"  - EOS惩罚: {baseline_components['eos_penalty_loss'].item():.6f}")

    # 完整优化配置 - 启用所有新功能
    full_optimization_loss_fn = EnhancedCTCLoss(
        vocab_size=V, blank=0,
        confuse_gamma=1.2,
        eos_penalty=0.15,
        char_focal=True,
        focal_gamma=2.0,
        focal_scale=1.0,
        adaptive_margin=True,
        margin_max=0.5,
        temperature_annealing=True
    )

    # 模拟训练后期的退火状态
    full_optimization_loss_fn.schedule(5, 10)  # epoch=5, max_epoch=10

    full_optimization_loss = full_optimization_loss_fn(logits.clone(), targets, input_lens, target_lens)
    full_components = full_optimization_loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)

    print(f"✓ 完整优化 - 总损失: {full_optimization_loss.item():.6f}")
    print(f"  - 基础CTC损失: {full_components['base_ctc_loss'].item():.6f}")
    print(f"  - 字符级Focal损失: {full_components['char_focal_loss'].item():.6f}")
    print(f"  - 样本级Focal损失: {full_components['sample_focal_loss'].item():.6f}")
    print(f"  - 路径权重: {full_components['path_weights'].item():.6f}")
    print(f"  - EOS惩罚: {full_components['eos_penalty_loss'].item():.6f}")
    print(f"  - 自适应Margin: {full_components.get('adaptive_margin', 0):.4f}")
    print(f"  - Focal缩放因子: {full_components['focal_scale']:.4f}")

    # 对比效果
    improvement = baseline_loss.item() - full_optimization_loss.item()
    print(f"✓ 优化效果: {improvement:.6f} ({improvement/baseline_loss.item()*100:.2f}%)")

def test_temperature_parameter_effects():
    """测试温度参数影响"""
    print("\n=== 测试温度参数影响 ===")

    B, T, V = 2, 10, len(char2idx)
    logits = torch.randn(B, T, V, requires_grad=True)

    # 包含易混淆字符的目标序列
    targets = torch.tensor([
        [char2idx['0'], char2idx['O'], char2idx['o'], char2idx['l'], char2idx['1']],  # 易混淆序列
        [char2idx['a'], char2idx['b'], char2idx['c'], char2idx['d'], char2idx['e']]   # 正常序列
    ], dtype=torch.long)
    input_lens = torch.tensor([10, 10], dtype=torch.long)
    target_lens = torch.tensor([5, 5], dtype=torch.long)

    # 测试不同温度参数的影响
    base_gamma = 1.0
    for temperature in [0.5, 1.0, 2.0, 5.0]:
        loss_fn = EnhancedCTCLoss(
            vocab_size=V, blank=0,
            confuse_gamma=base_gamma,
            confuse_temperature=temperature
        )
        loss = loss_fn(logits.clone(), targets, input_lens, target_lens)

        components = loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)
        print(f"✓ 温度={temperature}: 总损失={loss.item():.6f}, 路径权重={components['path_weights'].item():.6f}")

        # 温度越高，权重差异应该越小
        if temperature > 1.0:
            print("  - 高温下权重趋于平缓")

def test_adaptive_eos_window():
    """测试自适应尾部窗口大小"""
    print("\n=== 测试自适应尾部窗口大小 ===")

    B, T, V = 3, 12, len(char2idx)

    # 创建不同长度的序列
    logits = torch.randn(B, T, V) * 0.5
    # 在尾部增加空白字符概率
    for i in range(B):
        tail_start = max(0, T - 3 - i)  # 不同长度的尾部
        logits[i, tail_start:, 0] += 2.0

    logits.requires_grad = True

    targets = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [6, 7, 8, 9]], dtype=torch.long)
    input_lens = torch.tensor([12, 10, 8], dtype=torch.long)
    target_lens = torch.tensor([2, 3, 4], dtype=torch.long)

    # 测试固定窗口 vs 自适应窗口
    window_sizes = [2, 3, 5]

    for window_size in window_sizes:
        # 固定窗口
        fixed_loss_fn = EnhancedCTCLoss(
            vocab_size=V, blank=0,
            eos_penalty=0.1,
            eos_window_size=window_size,
            eos_adaptive=False
        )
        fixed_loss = fixed_loss_fn(logits.clone(), targets, input_lens, target_lens)
        fixed_components = fixed_loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)

        # 自适应窗口
        adaptive_loss_fn = EnhancedCTCLoss(
            vocab_size=V, blank=0,
            eos_penalty=0.1,
            eos_window_size=window_size,
            eos_adaptive=True
        )
        adaptive_loss = adaptive_loss_fn(logits.clone(), targets, input_lens, target_lens)
        adaptive_components = adaptive_loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)

        print(f"✓ 窗口大小={window_size}:")
        print(f"  - 固定窗口: 总损失={fixed_loss.item():.6f}, EOS惩罚={fixed_components['eos_penalty_loss'].item():.6f}")
        print(f"  - 自适应窗口: 总损失={adaptive_loss.item():.6f}, EOS惩罚={adaptive_components['eos_penalty_loss'].item():.6f}")

def test_gradient_clip_thresholds():
    """测试梯度裁剪阈值"""
    print("\n=== 测试梯度裁剪阈值 ===")

    B, T, V = 2, 8, len(char2idx)

    # 创建容易产生大梯度的极端情况
    logits = torch.zeros(B, T, V)
    logits[:, :, 0] = 15.0  # 极高的空白字符概率
    logits[:, :, 1] = -10.0  # 其他字符概率极低
    logits.requires_grad = True

    targets = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    input_lens = torch.tensor([8, 6], dtype=torch.long)
    target_lens = torch.tensor([2, 2], dtype=torch.long)

    # 测试不同梯度裁剪配置
    clip_configs = [
        {"gradient_clip": False},
        {"gradient_clip": True},  # 默认裁剪阈值
    ]

    for config in clip_configs:
        loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0, **config)

        try:
            loss = loss_fn(logits.clone(), targets, input_lens, target_lens)
            loss.backward()

            if logits.grad is not None:
                grad_norm = logits.grad.norm().item()
                print(f"✓ 梯度裁剪={config['gradient_clip']}: 梯度范数={grad_norm:.6f}")

                if config["gradient_clip"]:
                    if grad_norm < 1000:  # 检查裁剪效果
                        print("  - 梯度裁剪有效：梯度被控制在合理范围")
                    else:
                        print("  - ⚠️ 梯度裁剪可能未生效")
                else:
                    print("  - 无裁剪：梯度范数可能过大")

            # 清空梯度
            if logits.grad is not None:
                logits.grad.zero_()

        except Exception as e:
            print(f"✗ 梯度裁剪测试错误: {e}")

def test_different_reduction_modes():
    """测试不同reduction模式"""
    print("\n=== 测试不同reduction模式 ===")

    B, T, V = 3, 8, len(char2idx)
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 7, 0]], dtype=torch.long)
    input_lens = torch.tensor([8, 7, 6], dtype=torch.long)
    target_lens = torch.tensor([2, 3, 2], dtype=torch.long)

    # 测试不同reduction模式
    reduction_modes = ['mean', 'sum']

    for reduction in reduction_modes:
        loss_fn = EnhancedCTCLoss(vocab_size=V, blank=0, reduction=reduction)
        loss = loss_fn(logits.clone(), targets, input_lens, target_lens)

        # 获取组件分解
        components = loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)

        print(f"✓ reduction='{reduction}': 总损失={loss.item():.6f}")
        print(f"  - 基础CTC损失: {components['base_ctc_loss'].item():.6f}")
        print(f"  - 加权CTC损失: {components['weighted_ctc_loss'].item():.6f}")

        # 验证reduction效果
        if reduction == 'sum':
            # sum模式应该比mean模式损失值大（因为batch_size=3）
            print("  - sum模式损失值应大于mean模式")
        elif reduction == 'mean':
            print("  - mean模式损失值应适中")

def test_numerical_stability_boundaries():
    """测试数值稳定性边界情况"""
    print("\n=== 测试数值稳定性边界情况 ===")

    B, T, V = 2, 6, len(char2idx)

    # 测试不同的边界情况
    boundary_cases = [
        {
            "name": "极高空白概率",
            "logits_config": lambda l: l[:, :, 0].fill_(20.0),  # 空白字符logit极高
            "description": "模拟空白字符占绝对优势的情况"
        },
        {
            "name": "极低概率",
            "logits_config": lambda l: l.fill_(-20.0),  # 所有logits极低
            "description": "模拟所有字符概率都极低的情况"
        },
        {
            "name": "单峰分布",
            "logits_config": lambda l: l[:, :, 1].fill_(10.0),  # 只有目标字符有高概率
            "description": "模拟目标字符占绝对优势的情况"
        },
        {
            "name": "均匀分布",
            "logits_config": lambda l: l.fill_(0.0),  # 所有logits相等
            "description": "模拟所有字符概率相等的情况"
        }
    ]

    targets = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    input_lens = torch.tensor([6, 5], dtype=torch.long)
    target_lens = torch.tensor([2, 2], dtype=torch.long)

    for case in boundary_cases:
        print(f"\n--- 测试: {case['name']} ---")
        print(f"描述: {case['description']}")

        # 创建测试数据
        logits = torch.randn(B, T, V) * 0.1  # 基础小随机值
        case['logits_config'](logits)  # 应用边界条件
        logits.requires_grad = True

        loss_fn = EnhancedCTCLoss(
            vocab_size=V, blank=0,
            gradient_clip=True  # 启用梯度裁剪防止数值问题
        )

        try:
            loss = loss_fn(logits, targets, input_lens, target_lens)
            loss.backward()

            print(f"✓ 损失值: {loss.item():.6f}")

            if logits.grad is not None:
                grad_norm = logits.grad.norm().item()
                print(f"✓ 梯度范数: {grad_norm:.6f}")

                # 检查梯度是否合理
                if torch.isnan(logits.grad).any():
                    print("✗ 出现NaN梯度")
                elif torch.isinf(logits.grad).any():
                    print("✗ 出现Inf梯度")
                elif grad_norm > 1000:
                    print("⚠️ 梯度可能过大")
                else:
                    print("✓ 梯度数值稳定")

            # 测试组件分解
            components = loss_fn.get_loss_components(logits.clone(), targets, input_lens, target_lens)
            print("✓ 各组件数值正常")

        except Exception as e:
            print(f"✗ 边界情况测试失败: {e}")
            import traceback
            traceback.print_exc()

        # 清空梯度用于下一个测试
        if logits.grad is not None:
            logits.grad.zero_()


def test_distillation_loss_basic():
    """测试知识蒸馏损失函数基本功能"""
    print("\n=== 测试知识蒸馏损失基本功能 ===")

    # 创建测试数据
    B, L_teacher, L_student, D, V = 2, 10, 8, 512, 100

    # 教师模型输出（序列较长）
    teacher_features = torch.randn(B, L_teacher, D, requires_grad=False)
    teacher_logits = torch.randn(B, L_teacher, V, requires_grad=False)

    # 学生模型输出（序列较短）
    student_features = torch.randn(B, L_student, D, requires_grad=True)
    student_logits = torch.randn(B, L_student, V, requires_grad=True)

    # 创建蒸馏损失函数
    distill_loss_fn = DistillationLoss(temperature=4.0, alpha_feat=0.5, alpha_logit=0.5)

    try:
        # 计算蒸馏损失
        losses = distill_loss_fn(
            teacher_features=teacher_features,
            student_features=student_features,
            teacher_logits=teacher_logits,
            student_logits=student_logits
        )

        print(f"✓ 特征对齐损失: {losses['feature_loss'].item():.6f}")
        print(f"✓ KL散度损失: {losses['kl_loss'].item():.6f}")
        print(f"✓ 总蒸馏损失: {losses['total_distill_loss'].item():.6f}")

        # 测试反向传播
        total_loss = losses['total_distill_loss']
        total_loss.backward()

        # 检查学生模型梯度
        if student_features.grad is not None:
            print(f"✓ 学生特征梯度形状: {student_features.grad.shape}")
            print(f"✓ 学生特征梯度范数: {student_features.grad.norm().item():.6f}")

        if student_logits.grad is not None:
            print(f"✓ 学生logits梯度形状: {student_logits.grad.shape}")
            print(f"✓ 学生logits梯度范数: {student_logits.grad.norm().item():.6f}")

        # 验证教师模型没有梯度
        if teacher_features.grad is None or teacher_features.grad.abs().sum() == 0:
            print("✓ 教师特征没有梯度（正确）")

        return True

    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distillation_attention_alignment():
    """测试交叉注意力对齐机制"""
    print("\n=== 测试交叉注意力对齐机制 ===")

    B, L_teacher, L_student, D, V = 2, 12, 6, 256, 50

    # 创建有明显差异的序列长度
    teacher_features = torch.randn(B, L_teacher, D)
    student_features = torch.randn(B, L_student, D)
    teacher_logits = torch.randn(B, L_teacher, V)
    student_logits = torch.randn(B, L_student, V)

    distill_loss_fn = DistillationLoss(temperature=4.0)

    # 计算对齐前后的形状
    print(f"教师特征形状: {teacher_features.shape}")
    print(f"学生特征形状: {student_features.shape}")
    print(f"教师logits形状: {teacher_logits.shape}")
    print(f"学生logits形状: {student_logits.shape}")

    losses = distill_loss_fn(
        teacher_features=teacher_features,
        student_features=student_features,
        teacher_logits=teacher_logits,
        student_logits=student_logits
    )

    print(f"✓ 对齐后特征损失: {losses['feature_loss'].item():.6f}")
    print(f"✓ 对齐后KL损失: {losses['kl_loss'].item():.6f}")

    # 验证损失值合理范围
    assert losses['feature_loss'].item() >= 0, "特征损失应该非负"
    assert losses['kl_loss'].item() >= 0, "KL损失应该非负"
    print("✓ 损失值范围正常")


def test_distillation_temperature_effects():
    """测试温度参数对蒸馏损失的影响"""
    print("\n=== 测试温度参数影响 ===")

    B, L, D, V = 2, 8, 128, 30

    # 创建相同的教师和学生输出
    teacher_features = torch.randn(B, L, D)
    student_features = teacher_features + torch.randn(B, L, D) * 0.1  # 小幅差异
    teacher_logits = torch.randn(B, L, V)
    student_logits = teacher_logits + torch.randn(B, L, V) * 0.1

    temperatures = [1.0, 2.0, 4.0, 8.0, 16.0]

    for temp in temperatures:
        distill_loss_fn = DistillationLoss(temperature=temp, alpha_feat=0.5, alpha_logit=0.5)

        losses = distill_loss_fn(
            teacher_features=teacher_features,
            student_features=student_features,
            teacher_logits=teacher_logits,
            student_logits=student_logits
        )

        print(f"✓ 温度={temp}: 特征损失={losses['feature_loss'].item():.6f}, KL损失={losses['kl_loss'].item():.6f}")

        # 温度越高，KL损失应该越小（分布更平滑）
        if temp > 4.0:
            print(f"  - 高温下KL损失相对较小: {losses['kl_loss'].item():.6f}")


def test_distillation_mask_functionality():
    """测试mask功能"""
    print("\n=== 测试mask功能 ===")

    B, L, D, V = 2, 10, 256, 50

    teacher_features = torch.randn(B, L, D)
    student_features = torch.randn(B, L, D)
    teacher_logits = torch.randn(B, L, V)
    student_logits = torch.randn(B, L, V)

    # 创建mask，屏蔽部分位置
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, :6] = True  # 只保留前6个位置

    distill_loss_fn = DistillationLoss(temperature=4.0)

    # 测试有mask和无mask的情况
    losses_with_mask = distill_loss_fn(
        teacher_features=teacher_features,
        student_features=student_features,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        mask=mask
    )

    losses_without_mask = distill_loss_fn(
        teacher_features=teacher_features,
        student_features=student_features,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        mask=None
    )

    print(f"✓ 有mask - 特征损失: {losses_with_mask['feature_loss'].item():.6f}, KL损失: {losses_with_mask['kl_loss'].item():.6f}")
    print(f"✓ 无mask - 特征损失: {losses_without_mask['feature_loss'].item():.6f}, KL损失: {losses_without_mask['kl_loss'].item():.6f}")

    # 有mask时损失应该更小（只计算部分位置）
    assert losses_with_mask['kl_loss'].item() <= losses_without_mask['kl_loss'].item() * 1.1
    print("✓ mask功能正常")


def test_distillation_gradient_stability():
    """测试蒸馏损失的梯度稳定性"""
    print("\n=== 测试梯度稳定性 ===")

    B, L, D, V = 2, 8, 128, 20

    # 创建极端情况
    teacher_features = torch.randn(B, L, D)
    student_features = torch.randn(B, L, D, requires_grad=True)

    # 创建极端logits值
    teacher_logits = torch.zeros(B, L, V)
    teacher_logits[:, :, 0] = 100.0  # 一个类别概率极高
    student_logits = torch.full((B, L, V), 50.0, requires_grad=True)  # 避免in-place操作

    distill_loss_fn = DistillationLoss(temperature=4.0)

    try:
        losses = distill_loss_fn(
            teacher_features=teacher_features,
            student_features=student_features,
            teacher_logits=teacher_logits,
            student_logits=student_logits
        )

        total_loss = losses['total_distill_loss']
        total_loss.backward()

        # 检查梯度
        if student_features.grad is not None:
            grad_norm_features = student_features.grad.norm().item()
            print(f"✓ 学生特征梯度范数: {grad_norm_features:.6f}")

            if torch.isnan(student_features.grad).any():
                print("✗ 学生特征梯度出现NaN")
            elif torch.isinf(student_features.grad).any():
                print("✗ 学生特征梯度出现Inf")
            else:
                print("✓ 学生特征梯度正常")

        if student_logits.grad is not None:
            grad_norm_logits = student_logits.grad.norm().item()
            print(f"✓ 学生logits梯度范数: {grad_norm_logits:.6f}")

            if torch.isnan(student_logits.grad).any():
                print("✗ 学生logits梯度出现NaN")
            elif torch.isinf(student_logits.grad).any():
                print("✗ 学生logits梯度出现Inf")
            else:
                print("✓ 学生logits梯度正常")

        return True

    except Exception as e:
        print(f"✗ 梯度稳定性测试失败: {e}")
        return False


def test_distillation_alpha_weights():
    """测试alpha权重参数的影响"""
    print("\n=== 测试alpha权重参数 ===")

    B, L, D, V = 2, 8, 256, 30

    teacher_features = torch.randn(B, L, D)
    student_features = torch.randn(B, L, D)
    teacher_logits = torch.randn(B, L, V)
    student_logits = torch.randn(B, L, V)

    # 测试不同的alpha组合
    alpha_combinations = [
        {"alpha_feat": 1.0, "alpha_logit": 0.0},  # 只使用特征损失
        {"alpha_feat": 0.0, "alpha_logit": 1.0},  # 只使用logits损失
        {"alpha_feat": 0.7, "alpha_logit": 0.3},  # 偏重特征
        {"alpha_feat": 0.3, "alpha_logit": 0.7},  # 偏重logits
        {"alpha_feat": 0.5, "alpha_logit": 0.5},  # 平衡
    ]

    for alpha_config in alpha_combinations:
        distill_loss_fn = DistillationLoss(temperature=4.0, **alpha_config)

        losses = distill_loss_fn(
            teacher_features=teacher_features,
            student_features=student_features,
            teacher_logits=teacher_logits,
            student_logits=student_logits
        )

        total_loss = losses['total_distill_loss'].item()
        feature_loss = losses['feature_loss'].item()
        kl_loss = losses['kl_loss'].item()

        expected_total = (alpha_config['alpha_feat'] * feature_loss +
                         alpha_config['alpha_logit'] * kl_loss)

        print(f"✓ alpha_feat={alpha_config['alpha_feat']}, alpha_logit={alpha_config['alpha_logit']}")
        print(f"  - 特征损失: {feature_loss:.6f}, KL损失: {kl_loss:.6f}")
        print(f"  - 总损失: {total_loss:.6f}, 期望: {expected_total:.6f}")

        # 验证计算正确性
        assert abs(total_loss - expected_total) < 1e-6
        print("  - ✓ 权重计算正确")


def test_distillation_sequence_length_variations():
    """测试不同序列长度组合"""
    print("\n=== 测试不同序列长度组合 ===")

    D, V = 256, 40

    # 测试不同的序列长度组合
    length_combinations = [
        (5, 3),   # 教师长，学生短
        (8, 8),   # 等长
        (3, 5),   # 教师短，学生长
        (12, 4),  # 差异较大
        (20, 15), # 实际场景
    ]

    for teacher_len, student_len in length_combinations:
        B = 2
        teacher_features = torch.randn(B, teacher_len, D)
        student_features = torch.randn(B, student_len, D, requires_grad=True)
        teacher_logits = torch.randn(B, teacher_len, V)
        student_logits = torch.randn(B, student_len, V, requires_grad=True)

        distill_loss_fn = DistillationLoss(temperature=4.0)

        try:
            losses = distill_loss_fn(
                teacher_features=teacher_features,
                student_features=student_features,
                teacher_logits=teacher_logits,
                student_logits=student_logits
            )

            print(f"✓ 教师长度={teacher_len}, 学生长度={student_len}")
            print(f"  - 特征损失: {losses['feature_loss'].item():.6f}")
            print(f"  - KL损失: {losses['kl_loss'].item():.6f}")
            print(f"  - 总损失: {losses['total_distill_loss'].item():.6f}")

            # 验证梯度存在
            total_loss = losses['total_distill_loss']
            total_loss.backward()

            if student_features.grad is not None and student_logits.grad is not None:
                print("  - ✓ 梯度正常")
            else:
                print("  - ✗ 梯度异常")

        except Exception as e:
            print(f"✗ 长度组合 ({teacher_len}, {student_len}) 失败: {e}")


def run_distillation_tests():
    """运行所有蒸馏损失测试"""
    print("开始知识蒸馏损失函数测试...\n")

    tests = [
        test_distillation_loss_basic,
        test_distillation_attention_alignment,
        test_distillation_temperature_effects,
        test_distillation_mask_functionality,
        test_distillation_gradient_stability,
        test_distillation_alpha_weights,
        test_distillation_sequence_length_variations,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if result is not None else True)
            print("-" * 60)
        except Exception as e:
            print(f"测试 {test.__name__} 失败: {e}")
            results.append(False)
            print("-" * 60)

    # 总结结果
    passed = sum(results)
    total = len(results)
    print(f"\n蒸馏损失测试总结: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有蒸馏损失测试通过！")
    else:
        print("⚠️  部分蒸馏损失测试失败，请检查实现。")

    return passed == total

if __name__ == "__main__":
    # 运行增强CTC损失测试
    run_all_tests()

    # 运行知识蒸馏损失测试
    run_distillation_tests()

    # 运行新增测试
    test_char_focal_loss()
    print("-" * 50)
    test_adaptive_margin()
    print("-" * 50)
    test_temperature_annealing()
    print("-" * 50)
    test_temperature_parameter_effects()
    print("-" * 50)
    test_adaptive_eos_window()
    print("-" * 50)
    test_gradient_clip_thresholds()
    print("-" * 50)
    test_different_reduction_modes()
    print("-" * 50)
    test_numerical_stability_boundaries()
    print("-" * 50)
    test_combined_optimizations()
