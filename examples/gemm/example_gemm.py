"""
TileLang GEMM（通用矩阵乘法）示例程序
======================================

本示例展示了如何使用 TileLang 编写一个高性能的 CUDA GEMM 内核。
GEMM 是深度学习中最核心的计算操作之一，形式为：C = A @ B

矩阵维度说明：
    A: (M, K) —— 左矩阵，形状为 M 行 K 列
    B: (K, N) —— 右矩阵，形状为 K 行 N 列
    C: (M, N) —— 输出矩阵，形状为 M 行 N 列

计算原理：
    C[i][j] = sum(A[i][k] * B[k][j] for k in range(K))

本实现采用了经典的分块矩阵乘法（Tiled Matrix Multiplication）策略，
利用 CUDA 共享内存（Shared Memory）来减少全局内存访问延迟，
并通过流水线（Pipeline）技术隐藏内存加载的等待时间。
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    """
    构造分块 GEMM 计算内核函数。

    参数：
        M (int): 矩阵 A 的行数，也是输出矩阵 C 的行数
        N (int): 矩阵 B 的列数，也是输出矩阵 C 的列数
        K (int): 矩阵 A 的列数 / 矩阵 B 的行数（约化维度）
        block_M (int): 沿 M 维度的分块大小 —— 每个 CUDA 线程块负责计算的行数
        block_N (int): 沿 N 维度的分块大小 —— 每个 CUDA 线程块负责计算的列数
        block_K (int): 沿 K 维度的分块大小 —— 每次迭代加载的 K 方向切片大小
        dtype: 输入矩阵 A、B 的数据类型，默认 float16（半精度）
        accum_dtype: 累加器（累加计算）的数据类型，默认 float32（单精度）
                     使用更高精度的累加器可以避免浮点运算中的精度损失

    返回：
        gemm: 编译好的可调用内核函数
    """

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        """
        GEMM 原语函数定义。

        参数：
            A: 输入矩阵，形状 (M, K)，存储在全局内存（Global Memory）中
            B: 输入矩阵，形状 (K, N)，存储在全局内存中
            C: 输出矩阵，形状 (M, N)，结果写回全局内存
        """

        # ============================================================
        # 启动 CUDA Kernel（内核）
        # ============================================================
        # 网格配置（Grid Configuration）：
        #   - bx（block_x）：沿 N 轴的线程块索引，共 ceil(N/block_N) 个块
        #   - by（block_y）：沿 M 轴的线程块索引，共 ceil(M/block_M) 个块
        #   - threads=128：每个线程块包含 128 个线程
        #
        # 整体并行方式：将输出矩阵 C 划分为 (block_M × block_N) 的小块，
        # 每个线程块独立计算其中一个小块的最终结果。
        # ============================================================
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # --------------------------------------------------------
            # 分配共享内存（Shared Memory）
            # --------------------------------------------------------
            # 共享内存是片上内存（On-chip Memory），带宽远高于全局内存，
            # 且支持线程块内所有线程的低延迟广播访问。
            #
            # 策略：每个线程块先将需要用到的 A 和 B 的子块从全局内存
            # 加载到共享内存中，后续计算直接从共享内存读取数据。
            # --------------------------------------------------------

            # A_shared：缓存当前线程块所需的 A 矩阵子块，形状 (block_M, block_K)
            A_shared = T.alloc_shared((block_M, block_K), dtype)

            # B_shared：缓存当前线程块所需的 B 矩阵子块，形状 (block_K, block_N)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # --------------------------------------------------------
            # 分配寄存器/局部内存片段（Fragment）
            # --------------------------------------------------------
            # Fragment 通常映射到 GPU 寄存器文件（Register File），
            # 是速度最快的存储层级。每个线程拥有自己独立的 fragment 副本。
            #
            # 使用 accum_dtype（float32）作为累加精度，这是 GEMM 的标准做法：
            # 输入用 fp16 做乘法，累加用 fp32 避免精度丢失（Tensor Core 也采用此策略）。
            # --------------------------------------------------------
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # ============================================================
            # 初始化累加器：将 C_local 清零
            # ============================================================
            # 在开始累加之前，必须将局部结果缓冲区清零，
            # 否则累加结果会包含未初始化的脏数据。
            # ============================================================
            T.clear(C_local)

            # ============================================================
            # 分块循环（Tiled Loop）：沿 K 维度进行流水线迭代
            # ============================================================
            # 循环次数 = ceil(K / block_K)，即需要多少次分块才能覆盖整个 K 维度
            #
            # T.Pipelined（流水线循环）的关键作用：
            #   - num_stages=3：使用 3 级流水线（双缓冲 + 1 级计算）
            #   - 当当前 stage 正在用已加载的数据做 GEMM 计算时，
            #     下一个 stage 可以同时从全局内存预加载下一份数据到共享内存
            #   - 这种"计算与内存加载重叠"的技术可以有效隐藏全局内存的高延迟
            #
            # 直观理解：类似于 CPU 的预取（Prefetch），在用到数据之前提前加载
            # ============================================================
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # ---- 第 1 步：将 A 的子块从全局内存拷贝到共享内存 ----
                # 从 A 矩阵中取出当前位置的子块：
                #   - 行范围：[by * block_M, (by+1) * block_M) —— 当前线程块负责的 M 范围
                #   - 列范围：[k * block_K, (k+1) * block_K) —— 当前 K 迭代的分块范围
                T.copy(A[by * block_M, k * block_K], A_shared)

                # ---- 第 2 步：将 B 的子块从全局内存拷贝到共享内存 ----
                # 从 B 矩阵中取出当前位置的子块：
                #   - 行范围：[k * block_K, (k+1) * block_K) —— 当前 K 迭代的分块范围
                #   - 列范围：[bx * block_N, (bx+1) * block_N) —— 当前线程块负责的 N 范围
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # ---- 第 3 步：执行分块矩阵乘法并累加 ----
                # 计算：C_local += A_shared @ B_shared
                # 这一步操作的是已经加载到共享内存/寄存器中的数据，
                # 充分利用了 GPU 片上存储的高带宽特性。
                # 如果硬件支持 Tensor Core，这里可能会自动映射到 HMMA 指令。
                T.gemm(A_shared, B_shared, C_local)

            # ============================================================
            # 将计算结果从局部内存写回全局内存
            # ============================================================
            # 经过所有 K 分块的迭代后，C_local 中存储了完整的
            # C[by*block_M:(by+1)*block_M, bx*block_N:(bx+1)*block_N] 子块结果。
            # 最后将其写回输出矩阵 C 在全局内存中的对应位置。
            # ============================================================
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main(warmup=10, repeat=100):
    """
    主函数：演示 GEMM 内核的完整使用流程，包括：
      1. 编译内核
      2. 准备输入数据
      3. 执行计算 & 正确性验证
      4. 打印生成的 CUDA 源码
      5. 预热（Warmup）—— 让 GPU 达到稳定工作状态
      6. 重复执行区域 —— 配合 nsys profile 采集 kernel timeline
      7. 性能基准测试（Benchmark）

    参数：
        warmup (int): 预热轮次，让 GPU 达到稳定工作状态（默认 10）
        repeat (int): 重复执行次数（默认 100）
                   - 通过 nsys profile 运行时，此区域会被完整采集到 timeline 中
                   - 建议设为较大值以获得更完整的 GPU activity 视图

    使用方式（Nsight Systems 剖析）：
        在 Ray 集群上提交时加 --environment 或通过 nsys 包装：

        ray job submit --address='http://<host>:8420' \
            --working-dir examples/gemm \
            -- nysys profile -o gemm_nsys_report --trace=cuda,nvtx --stats=true \
                python example_gemm.py

        或者设置环境变量启用 nsys 模式：
        NSYS_PROFILE=1 ray job submit ...
    """

    import os

    # 是否处于 nsys profile 模式（通过环境变量检测）
    nsys_mode = os.environ.get("NSYS_PROFILE", "0") == "1"

    # ================================================================
    # 第 1 步：编译 GEMM 内核
    # ================================================================
    # 参数含义：M=1024, N=1024, K=1024 —— 计算 1024×1024 的矩阵乘法
    #           block_M=128, block_N=128 —— 每个线程块计算 128×128 的输出子块
    #           block_K=32 —— 每次 K 迭代加载 32 列 A 和 32 行 B
    #
    # tilelang.jit 装饰器会将上面的 Python 描述编译成优化的 CUDA 内核
    # ================================================================
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch

    # ================================================================
    # 第 2 步：准备测试数据
    # ================================================================
    # 生成两个 1024×1024 的随机矩阵，使用正态分布随机初始化
    # .cuda()   —— 将数据放到 GPU 显存上
    # .half()   —— 转换为 float16（半精度浮点数）
    # ================================================================
    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    # ================================================================
    # 第 3 步：执行 TileLang 编译的 GEMM 内核（单次运行）
    # ================================================================
    # 直接像调用普通 Python 函数一样调用编译好的内核
    # 内部会自动启动 CUDA Kernel 进行 GPU 并行计算
    # ================================================================
    c = kernel(a, b)

    # ================================================================
    # 第 4 步：计算参考结果（使用 PyTorch 内置的 matmul 作为对照）
    # ================================================================
    # PyTorch 的 @ 操作符会调用高度优化的 cuBLAS 库进行矩阵乘法
    # 我们用它作为"金标准"来验证自定义内核的正确性
    # ================================================================
    ref_c = a @ b

    # 打印计算结果供人工检查
    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    # ================================================================
    # 正确性验证
    # ================================================================
    # rtol=1e-2（相对容差 1%）和 atol=1e-2（绝对容差 0.01）
    # 由于使用了 float16 输入 + float32 累加，数值误差在可控范围内
    # 如果断言失败，会抛出异常提示计算结果不匹配
    # ================================================================
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # ================================================================
    # 第 5 步：查看生成的 CUDA 源代码
    # ================================================================
    # TileLang 将高层 Python 描述编译为底层 CUDA C++ 代码。
    # 打印生成的源码有助于理解编译器的优化过程和调试问题。
    # ================================================================
    print("CUDA Source:")
    print(kernel.get_kernel_source())

    # ================================================================
    # 第 6 步：预热（Warmup）
    # ================================================================
    # 预热的作用：
    #   1. 让 GPU 从低功耗状态提升到稳定工作频率（Boost 频率）
    #   2. 预热 CUDA Context、驱动程序和运行时开销
    #   3. 对于 nsys profile 来说，预热阶段的数据不会被纳入正式采集范围
    #          避免 JIT 编译、首次内存分配等一次性开销干扰分析结果
    # ================================================================
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        kernel(a, b)
    # 同步 CUDA 流，确保所有预热 kernel 已完成
    torch.cuda.synchronize()
    print("Warmup done.")

    # ================================================================
    # 第 7 步：nsys 剖析区域 / 重复执行
    # ================================================================
    # 这段循环是 nsys profile 的核心关注区域：
    #   - nsys 会记录每次 kernel launch 到完成的完整时间线
    #   - 可以在 Nsight Systems GUI 中看到每个 kernel 的：
    #       * 执行时长（GPU 时间 vs Wall Clock 时间）
    #       * SM 利用率、Tensor Core 占用率
    #       * 内存读写带宽（Memory Throughput）
    #       * TMA 异步拷贝与 WGMMA 计算的重叠情况
    #       * 流水线各 stage 的并行效率
    #
    # 使用 torch.cuda.nvtx.mark 标记区域边界，方便在 nsys 中定位
    # ================================================================
    try:
        # NVTX（NVIDIA Tools Extension）用于在 nsys timeline 中插入自定义标记
        # 这样可以在 GUI 中快速定位到我们关心的代码区间
        import torch.cuda.nvtx as nvtx
    except ImportError:
        nvtx = None  # 环境不支持 NVTX 时静默跳过

    if nvtx is not None:
        nvtx.range_push("tilelang_gemm_repeat")  # 开始 NVTX 范围

    print(f"Running repeat region ({repeat} iterations)...")
    for i in range(repeat):
        if nvtx is not None and i % 10 == 0:
            # 每 10 次迭代打一个标记，方便在 timeline 中定位位置
            nvtx.mark(f"iteration_{i}")
        kernel(a, b)

    # 同步确保所有 kernel 执行完毕后再结束采集
    torch.cuda.synchronize()

    if nvtx is not None:
        nvtx.range_pop()  # 结束 NVTX 范围
    print("Repeat region done.")

    # ================================================================
    # 第 8 步：性能基准测试（Benchmark）
    # ================================================================
    # get_profiler()：获取内核的性能分析器
    # do_bench(backend="cupti")：使用 CUPTI（CUDA Performance Tools Interface）
    #   后端进行精确的 GPU 性能测量，返回平均执行延迟（单位：毫秒）
    #   CUPTI 比单纯的 CUDA Event 计时更精确，能排除 CUDA API 调用的开销
    # ================================================================
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()  # 备选：使用默认的后端计时方式
    print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    """
    回归性能测试函数。

    用于性能回归检测：在持续集成（CI）或日常测试中，
    定期运行此函数来监控 GEMM 内核的性能是否出现意外退化。

    返回：
        float: 内核的平均执行延迟（毫秒）
    """
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TileLang GEMM 示例")
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="预热轮次（默认: 10），让 GPU 达到稳定工作状态",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="重复执行次数（默认: 100），nsys 会采集此区域的 kernel timeline",
    )
    args = parser.parse_args()

    main(warmup=args.warmup, repeat=args.repeat)


"""
ray job submit --address='http://33.29.18.248:8420' --working-dir examples/gemm -- python example_gemm.py
ray job submit --address='http://33.29.18.248:8420' --working-dir examples/gemm -- nsys profile -o gemm_nsys_report --trace=cuda,nvtx --stats=true -- python example_gemm.py --warmup 10 --repeat 100

nsys export -t sqlite -f true -o gemm_nsys_export.sqlite gemm_nsys_report.nsys-rep
sqlite3 -header -column ~/Downloads/gemm_nsys_export.sqlite " SELECT s.value AS kernel_name, COUNT(*) AS instances, ROUND(AVG(k.end - k.start), 2) AS avg_us, ROUND(MIN(k.end - k.start), 2) AS min_us, ROUND(MAX(k.end - k.start), 2) AS max_us, ROUND(SUM(k.end - k.start) / 1000.0, 2) AS total_ms FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id GROUP BY k.shortName ORDER BY total_ms DESC LIMIT 15; "
sqlite3 -header -column ~/Downloads/gemm_nsys_export.sqlite "SELECT k.start, k.end, (k.end-k.start) AS dur_us FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE k.shortName=(SELECT id FROM StringIds WHERE value='gemm_kernel') ORDER BY k.start LIMIT 20;"
sqlite3 -header -column ~/Downloads/gemm_nsys_export.sqlite "SELECT s.value AS nvtx_range, ROUND((n.end-n.start)/1000.0,3) AS dur_ms FROM NVTX_EVENTS n JOIN StringIds s ON n.text=s.id ORDER BY n.start;" 2>&1
sqlite3 -header -column ~/Downloads/gemm_nsys_export.sqlite "PRAGMA table_info(NVTX_EVENTS);" 2>&1
sqlite3 -header -column ~/Downloads/gemm_nsys_export.sqlite "SELECT text, eventType, ROUND((end-start)/1000.0,3) AS dur_ms FROM NVTX_EVENTS WHERE text IS NOT NULL AND text != '' ORDER BY start;" 2>&1
sqlite3 -header -column ~/Downloads/gemm_nsys_export.sqlite "SELECT s.value AS api_name, COUNT(*) AS calls, ROUND(AVG(r.end-r.start),1) AS avg_ns, ROUND(MAX(r.end-r.start),1) AS max_ns FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.shortName=s.id GROUP BY r.shortName ORDER BY SUM(r.end-r.start) DESC LIMIT 15;" 2>&1
sqlite3 -header -column ~/Downloads/gemm_nsys_export.sqlite "PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME);" 2>&1
sqlite3 -header -column ~/Downloads/gemm_nsys_export.sqlite "SELECT s.value AS api_name, COUNT(*) AS calls, ROUND(AVG(r.end-r.start),1) AS avg_ns FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id GROUP BY r.nameId ORDER BY SUM(r.end-r.start) DESC LIMIT 15;" 2>&1

open "/Applications/NVIDIA Nsight Systems.app" ~/Downloads/gemm_nsys_report.nsys-rep 2>&1
sqlite3 -header -column ~/Downloads/gemm_nsys_export.sqlite " SELECT s.value AS kernel_name, COUNT(*) AS instances, ROUND(AVG(k.end - k.start), 2) AS avg_us, ROUND(MIN(k.end - k.start), 2) AS min_us, ROUND(MAX(k.end - k.start), 2) AS max_us, ROUND(SUM(k.end - k.start) / 1000.0, 2) AS total_ms FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id GROUP BY k.shortName ORDER BY total_ms DESC LIMIT 15; "
"""
