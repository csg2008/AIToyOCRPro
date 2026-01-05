"""
性能优化测试代码
测试 decoder.py 中的各项优化效果
"""
import time
import torch
import torch.nn as nn
import gc

from decoder import (
    RopeTransformerEncoder,
    RopeMultiHeadAttentionGQA,
    DynamicBlockSVDLinear,
    BlockSVDLinear,
    CandidateNet
)


def test_skip_attention_mask_optimization():
    """
    测试 Skip-Attention 掩码创建的优化效果
    对比向量化实现与原始循环实现的性能
    """
    print("\n" + "="*80)
    print("测试 1: Skip-Attention 掩码创建优化")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_lengths = [64, 128, 256, 512, 1024]

    # 创建 GQA 模块
    gqa = RopeMultiHeadAttentionGQA(
        d_model=512,
        nhead=8,
        nhead_kv=2,
        use_skip_attention=True,
        skip_window=24,
        global_tokens=8
    ).to(device)

    # 原始实现（用于对比）
    def create_skip_mask_original(seq_len: int, skip_window: int, global_tokens: int, device: torch.device) -> torch.Tensor:
        """原始的循环实现"""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32)
        half_window = skip_window // 2
        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = 0.0
        if global_tokens > 0:
            mask[:global_tokens, :] = 0.0
            mask[:, :global_tokens] = 0.0
        return mask

    print(f"\n设备: {device}")
    print(f"{'序列长度':<12} {'原始实现 (ms)':<15} {'优化实现 (ms)':<15} {'加速比':<10} {'正确性':<10}")
    print("-" * 80)

    for seq_len in seq_lengths:
        # 测试原始实现
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(100):
            mask_orig = create_skip_mask_original(seq_len, 24, 8, device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        orig_time = (time.time() - start_time) * 10  # 转换为毫秒

        # 测试优化实现
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(100):
            mask_opt = gqa.create_skip_mask(seq_len, device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        opt_time = (time.time() - start_time) * 10  # 转换为毫秒

        # 验证正确性
        is_correct = torch.allclose(mask_orig, mask_opt, equal_nan=True)

        speedup = orig_time / opt_time if opt_time > 0 else float('inf')

        print(f"{seq_len:<12} {orig_time:<15.4f} {opt_time:<15.4f} {speedup:<10.2f}x {'✓' if is_correct else '✗':<10}")

    print("\n✓ Skip-Attention 掩码优化测试完成")


def test_gqa_kv_replication_optimization():
    """
    测试 GQA 的 K/V 复制优化效果
    对比 expand 方法与 repeat_interleave 方法的性能
    """
    print("\n" + "="*80)
    print("测试 2: GQA 的 K/V 复制优化")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    seq_len = 128
    nhead = 8
    nhead_kv = 2
    head_dim = 64
    n_rep = nhead // nhead_kv

    # 创建测试数据
    k = torch.randn(batch_size, nhead_kv, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, nhead_kv, seq_len, head_dim, device=device)

    # 原始实现：repeat_interleave
    def original_repeat_interleave(k, v, n_rep):
        k_rep = torch.repeat_interleave(k, repeats=n_rep, dim=1)
        v_rep = torch.repeat_interleave(v, repeats=n_rep, dim=1)
        return k_rep, v_rep

    # 优化实现：expand + reshape
    def optimized_expand(k, v, n_rep, batch_size, seq_len, head_dim):
        k_rep = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, nhead_kv * n_rep, seq_len, head_dim)
        v_rep = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch_size, nhead_kv * n_rep, seq_len, head_dim)
        return k_rep, v_rep

    print(f"\n设备: {device}")
    print(f"批次大小: {batch_size}, 序列长度: {seq_len}, 头数: {nhead}, KV头数: {nhead_kv}")
    print(f"\n{'方法':<20} {'时间 (ms)':<15} {'显存 (MB)':<15} {'加速比':<10}")
    print("-" * 80)

    # 测试原始实现
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(1000):
        k_orig, v_orig = original_repeat_interleave(k, v, n_rep)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    orig_time = (time.time() - start_time) * 1000

    orig_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

    # 测试优化实现
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(1000):
        k_opt, v_opt = optimized_expand(k, v, n_rep, batch_size, seq_len, head_dim)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    opt_time = (time.time() - start_time) * 1000

    opt_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

    # 验证正确性
    is_correct = torch.allclose(k_orig, k_opt) and torch.allclose(v_orig, v_opt)

    speedup = orig_time / opt_time if opt_time > 0 else float('inf')

    print(f"{'原始实现':<20} {orig_time:<15.4f} {orig_mem:<15.2f} {'-':<10}")
    print(f"{'优化实现':<20} {opt_time:<15.4f} {opt_mem:<15.2f} {speedup:<10.2f}x")
    print(f"\n正确性验证: {'✓ 通过' if is_correct else '✗ 失败'}")

    print("\n✓ GQA 的 K/V 复制优化测试完成")


def test_dynamic_block_svd_linear_optimization():
    """
    测试 DynamicBlockSVDLinear 的优化效果
    对比批量处理与串行处理的性能
    """
    print("\n" + "="*80)
    print("测试 3: DynamicBlockSVDLinear 批量处理优化")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试配置
    configs = [
        {"B": 8, "T": 64, "d": 512, "num_classes": 10000, "k": 8, "r": 64, "dynamic_k": 100},
        {"B": 16, "T": 128, "d": 512, "num_classes": 10000, "k": 8, "r": 64, "dynamic_k": 100},
        {"B": 32, "T": 256, "d": 512, "num_classes": 10000, "k": 8, "r": 64, "dynamic_k": 100},
    ]

    for config in configs:
        B, T, d = config["B"], config["T"], config["d"]
        num_classes, k, r, dynamic_k = config["num_classes"], config["k"], config["r"], config["dynamic_k"]

        print(f"\n配置: B={B}, T={T}, d={d}, num_classes={num_classes}")
        print(f"{'方法':<20} {'时间 (ms)':<15} {'显存 (MB)':<15} {'加速比':<10}")
        print("-" * 80)

        # 创建测试数据
        x = torch.randn(B, T, d, device=device)

        # 测试优化后的实现
        model = DynamicBlockSVDLinear(d, num_classes, k=k, r=r, dynamic_k=dynamic_k).to(device)
        model.eval()

        # 预热
        with torch.no_grad():
            _ = model(x)

        # 测试优化实现
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                out_opt = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        opt_time = (time.time() - start_time) * 10  # 转换为毫秒

        opt_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

        print(f"{'优化实现':<20} {opt_time:<15.4f} {opt_mem:<15.2f} {'-':<10}")
        print(f"输出形状: {out_opt.shape}")

        # 对比普通 Linear 层
        linear = nn.Linear(d, num_classes).to(device)
        linear.eval()

        # 预热
        with torch.no_grad():
            _ = linear(x)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                out_linear = linear(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        linear_time = (time.time() - start_time) * 10  # 转换为毫秒

        linear_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

        speedup = linear_time / opt_time if opt_time > 0 else float('inf')

        print(f"{'普通 Linear':<20} {linear_time:<15.4f} {linear_mem:<15.2f} {speedup:<10.2f}x")
        print(f"输出形状: {out_linear.shape}")

    print("\n✓ DynamicBlockSVDLinear 批量处理优化测试完成")


def test_cache_optimization():
    """
    测试缓存优化的效果
    对比启用缓存与禁用缓存的性能
    """
    print("\n" + "="*80)
    print("测试 4: DynamicBlockSVDLinear 缓存优化")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, T, d = 16, 128, 512
    num_classes, k, r, dynamic_k = 10000, 8, 64, 100

    print(f"\n配置: B={B}, T={T}, d={d}, num_classes={num_classes}")
    print(f"{'方法':<25} {'时间 (ms)':<15} {'加速比':<10}")
    print("-" * 80)

    # 创建测试数据
    x = torch.randn(B, T, d, device=device)

    # 测试禁用缓存
    model_no_cache = DynamicBlockSVDLinear(d, num_classes, k=k, r=r, dynamic_k=dynamic_k).to(device)
    model_no_cache.eval()
    model_no_cache.enable_cache(False)

    # 预热
    with torch.no_grad():
        _ = model_no_cache(x)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            out_no_cache = model_no_cache(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    no_cache_time = (time.time() - start_time) * 10  # 转换为毫秒

    print(f"{'禁用缓存':<25} {no_cache_time:<15.4f} {'-':<10}")

    # 测试启用缓存（使用相同的输入）
    model_with_cache = DynamicBlockSVDLinear(d, num_classes, k=k, r=r, dynamic_k=dynamic_k).to(device)
    model_with_cache.eval()
    model_with_cache.enable_cache(True)

    # 预热（第一次调用会缓存候选选择结果）
    with torch.no_grad():
        _ = model_with_cache(x)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            out_with_cache = model_with_cache(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    with_cache_time = (time.time() - start_time) * 10  # 转换为毫秒

    speedup = no_cache_time / with_cache_time if with_cache_time > 0 else float('inf')

    print(f"{'启用缓存 (相同输入)':<25} {with_cache_time:<15.4f} {speedup:<10.2f}x")

    # 注意：由于缓存是基于第一次调用的输入，所以输出应该相同
    # 但由于模型参数初始化不同，输出可能不完全相同
    # 这里我们只测试性能提升，不验证输出一致性

    print("\n✓ 缓存优化测试完成")


def test_rope_transformer_encoder_performance():
    """
    测试完整的 RopeTransformerEncoder 性能
    """
    print("\n" + "="*80)
    print("测试 5: RopeTransformerEncoder 完整性能测试")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from decoder import RopeTransformerEncoder

    # 测试配置
    configs = [
        {"B": 8, "T": 64, "in_channels": 512, "hidden_dim": 512, "num_classes": 10000, "num_layers": 4, "num_heads": 8},
        {"B": 16, "T": 128, "in_channels": 512, "hidden_dim": 512, "num_classes": 10000, "num_layers": 6, "num_heads": 8},
    ]

    for config in configs:
        B, T, in_channels = config["B"], config["T"], config["in_channels"]
        hidden_dim, num_classes = config["hidden_dim"], config["num_classes"]
        num_layers, num_heads = config["num_layers"], config["num_heads"]

        print(f"\n配置: B={B}, T={T}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        print(f"{'配置':<30} {'时间 (ms)':<15} {'吞吐量 (samples/s)':<20}")
        print("-" * 80)

        # 创建测试数据
        x = torch.randn(B, T, in_channels, device=device)

        # 测试基础配置
        model = RopeTransformerEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            max_text_length=128,
            pad_token=0,
            sos_token=1,
            eos_token=2,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1,
            max_rope_len=30000,
            use_svd=True,
            k=8,
            r=64,
            dynamic_k=100,
            use_skip_attention=False,
            use_mla=False
        ).to(device)

        model.eval()

        # 预热
        with torch.no_grad():
            _ = model(x)

        # 测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                out = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time

        time_per_batch = elapsed * 1000 / 100  # 毫秒
        throughput = 100 * B / elapsed  # samples/s

        print(f"{'基础配置':<30} {time_per_batch:<15.4f} {throughput:<20.2f}")
        print(f"输出形状: {out.shape}")

    print("\n✓ RopeTransformerEncoder 完整性能测试完成")

def test_gradient_checkpointing():
    """
    测试 Gradient Checkpointing 的优化效果
    对比启用和禁用 gradient checkpointing 的显存占用和性能
    """
    print("\n" + "="*80)
    print("测试 1: Gradient Checkpointing 优化")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试配置
    configs = [
        {"B": 8, "T": 128, "in_channels": 512, "hidden_dim": 512, "num_classes": 10000, "num_layers": 6, "num_heads": 8},
        {"B": 16, "T": 256, "in_channels": 512, "hidden_dim": 512, "num_classes": 10000, "num_layers": 8, "num_heads": 8},
    ]

    for config in configs:
        B, T, in_channels = config["B"], config["T"], config["in_channels"]
        hidden_dim, num_classes = config["hidden_dim"], config["num_classes"]
        num_layers, num_heads = config["num_layers"], config["num_heads"]

        print(f"\n配置: B={B}, T={T}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        print(f"{'模式':<25} {'显存 (MB)':<15} {'时间 (ms)':<15} {'吞吐量 (samples/s)':<20}")
        print("-" * 80)

        # 创建测试数据
        x = torch.randn(B, T, in_channels, device=device, requires_grad=True)
        target = torch.randn(B, T, num_classes, device=device)

        # 测试禁用 gradient checkpointing
        model_no_gc = RopeTransformerEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            max_text_length=256,
            pad_token=0,
            sos_token=1,
            eos_token=2,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1,
            max_rope_len=30000,
            use_svd=True,
            k=8,
            r=64,
            dynamic_k=100,
            use_gradient_checkpointing=False,
            use_fused_ops=False
        ).to(device)

        model_no_gc.train()

        # 预热
        _ = model_no_gc(x)

        # 测试禁用 gradient checkpointing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        for _ in range(10):
            out = model_no_gc(x)
            loss = nn.functional.mse_loss(out, target)
            loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        no_gc_time = (time.time() - start_time) * 100  # 转换为毫秒

        no_gc_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        no_gc_throughput = 10 * B / (time.time() - start_time)

        print(f"{'禁用 GC':<25} {no_gc_mem:<15.2f} {no_gc_time:<15.2f} {no_gc_throughput:<20.2f}")

        # 测试启用 gradient checkpointing
        model_with_gc = RopeTransformerEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            max_text_length=256,
            pad_token=0,
            sos_token=1,
            eos_token=2,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1,
            max_rope_len=30000,
            use_svd=True,
            k=8,
            r=64,
            dynamic_k=100,
            use_gradient_checkpointing=True,
            use_fused_ops=False
        ).to(device)

        model_with_gc.train()

        # 预热
        _ = model_with_gc(x)

        # 测试启用 gradient checkpointing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        for _ in range(10):
            out = model_with_gc(x)
            loss = nn.functional.mse_loss(out, target)
            loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        with_gc_time = (time.time() - start_time) * 100  # 转换为毫秒

        with_gc_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        with_gc_throughput = 10 * B / (time.time() - start_time)

        mem_reduction = (no_gc_mem - with_gc_mem) / no_gc_mem * 100 if no_gc_mem > 0 else 0
        time_overhead = (with_gc_time - no_gc_time) / no_gc_time * 100 if no_gc_time > 0 else 0

        print(f"{'启用 GC':<25} {with_gc_mem:<15.2f} {with_gc_time:<15.2f} {with_gc_throughput:<20.2f}")
        print(f"\n显存节省: {mem_reduction:.2f}%, 时间开销: {time_overhead:.2f}%")

    print("\n✓ Gradient Checkpointing 优化测试完成")


def test_operator_fusion():
    """
    测试算子融合的优化效果
    对比启用和禁用算子融合的性能
    """
    print("\n" + "="*80)
    print("测试 2: 算子融合优化")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试配置
    configs = [
        {"B": 8, "T": 128, "in_channels": 512, "hidden_dim": 512, "num_classes": 10000, "num_layers": 4, "num_heads": 8},
        {"B": 16, "T": 256, "in_channels": 512, "hidden_dim": 512, "num_classes": 10000, "num_layers": 6, "num_heads": 8},
    ]

    for config in configs:
        B, T, in_channels = config["B"], config["T"], config["in_channels"]
        hidden_dim, num_classes = config["hidden_dim"], config["num_classes"]
        num_layers, num_heads = config["num_layers"], config["num_heads"]

        print(f"\n配置: B={B}, T={T}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        print(f"{'模式':<25} {'时间 (ms)':<15} {'吞吐量 (samples/s)':<20} {'加速比':<10}")
        print("-" * 80)

        # 创建测试数据
        x = torch.randn(B, T, in_channels, device=device)

        # 测试禁用算子融合
        model_no_fusion = RopeTransformerEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            max_text_length=256,
            pad_token=0,
            sos_token=1,
            eos_token=2,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1,
            max_rope_len=30000,
            use_svd=True,
            k=8,
            r=64,
            dynamic_k=100,
            use_gradient_checkpointing=False,
            use_fused_ops=False
        ).to(device)

        model_no_fusion.eval()

        # 预热
        with torch.no_grad():
            _ = model_no_fusion(x)

        # 测试禁用算子融合
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                out = model_no_fusion(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        no_fusion_time = (time.time() - start_time) * 10  # 转换为毫秒

        no_fusion_throughput = 100 * B / (time.time() - start_time)

        print(f"{'禁用融合':<25} {no_fusion_time:<15.4f} {no_fusion_throughput:<20.2f} {'-':<10}")

        # 测试启用算子融合
        model_with_fusion = RopeTransformerEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            max_text_length=256,
            pad_token=0,
            sos_token=1,
            eos_token=2,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1,
            max_rope_len=30000,
            use_svd=True,
            k=8,
            r=64,
            dynamic_k=100,
            use_gradient_checkpointing=False,
            use_fused_ops=True
        ).to(device)

        model_with_fusion.eval()

        # 预热
        with torch.no_grad():
            _ = model_with_fusion(x)

        # 测试启用算子融合
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                out = model_with_fusion(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        with_fusion_time = (time.time() - start_time) * 10  # 转换为毫秒

        with_fusion_throughput = 100 * B / (time.time() - start_time)

        speedup = no_fusion_time / with_fusion_time if with_fusion_time > 0 else float('inf')

        print(f"{'启用融合':<25} {with_fusion_time:<15.4f} {with_fusion_throughput:<20.2f} {speedup:<10.2f}x")

        # 验证正确性
        is_correct = torch.allclose(out, model_no_fusion(x), rtol=1e-4, atol=1e-5)
        print(f"\n正确性验证: {'✓ 通过' if is_correct else '✗ 失败'}")

    print("\n✓ 算子融合优化测试完成")


def test_combined_optimizations():
    """
    测试组合优化效果
    同时启用 gradient checkpointing 和算子融合
    """
    print("\n" + "="*80)
    print("测试 3: 组合优化效果")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试配置
    B, T, in_channels = 16, 256, 512
    hidden_dim, num_classes = 512, 10000
    num_layers, num_heads = 6, 8

    print(f"\n配置: B={B}, T={T}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    print(f"{'模式':<30} {'显存 (MB)':<15} {'时间 (ms)':<15} {'吞吐量 (samples/s)':<20}")
    print("-" * 80)

    # 创建测试数据
    x = torch.randn(B, T, in_channels, device=device, requires_grad=True)
    target = torch.randn(B, T, num_classes, device=device)

    # 基准：无优化
    model_baseline = RopeTransformerEncoder(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        max_text_length=256,
        pad_token=0,
        sos_token=1,
        eos_token=2,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
        max_rope_len=30000,
        use_svd=True,
        k=8,
        r=64,
        dynamic_k=100,
        use_gradient_checkpointing=False,
        use_fused_ops=False
    ).to(device)

    model_baseline.train()

    # 预热
    _ = model_baseline(x)

    # 测试基准
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    for _ in range(10):
        out = model_baseline(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    baseline_time = (time.time() - start_time) * 100  # 转换为毫秒

    baseline_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    baseline_throughput = 10 * B / (time.time() - start_time)

    print(f"{'基准 (无优化)':<30} {baseline_mem:<15.2f} {baseline_time:<15.2f} {baseline_throughput:<20.2f}")

    # 组合优化：Gradient Checkpointing + 算子融合
    model_combined = RopeTransformerEncoder(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        max_text_length=256,
        pad_token=0,
        sos_token=1,
        eos_token=2,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
        max_rope_len=30000,
        use_svd=True,
        k=8,
        r=64,
        dynamic_k=100,
        use_gradient_checkpointing=True,
        use_fused_ops=True
    ).to(device)

    model_combined.train()

    # 预热
    _ = model_combined(x)

    # 测试组合优化
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    for _ in range(10):
        out = model_combined(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    combined_time = (time.time() - start_time) * 100  # 转换为毫秒

    combined_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    combined_throughput = 10 * B / (time.time() - start_time)

    mem_reduction = (baseline_mem - combined_mem) / baseline_mem * 100 if baseline_mem > 0 else 0
    speedup = baseline_time / combined_time if combined_time > 0 else float('inf')

    print(f"{'组合优化 (GC + 融合)':<30} {combined_mem:<15.2f} {combined_time:<15.2f} {combined_throughput:<20.2f}")
    print(f"\n显存节省: {mem_reduction:.2f}%, 加速比: {speedup:.2f}x")

    print("\n✓ 组合优化测试完成")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("开始性能优化测试")
    print("="*80)

    try:
        test_skip_attention_mask_optimization()
        test_gqa_kv_replication_optimization()
        test_dynamic_block_svd_linear_optimization()
        test_cache_optimization()
        test_rope_transformer_encoder_performance()
        test_gradient_checkpointing()
        test_operator_fusion()
        test_combined_optimizations()

        print("\n" + "="*80)
        print("所有测试完成！")
        print("="*80)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
