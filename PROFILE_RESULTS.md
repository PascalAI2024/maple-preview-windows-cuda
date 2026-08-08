# Maple pp512 Profiling Results

## Method

Target: `C:/Users/pasca/dev/maple/fork` (local build of maple-tq2_0, patches 0001-0004 applied).
Binary: `build-dequant-fix/bin/llama-bench.exe` (Release, sm_89, CUDA 12.8).
GPU: NVIDIA GeForce RTX 4080 SUPER (16 GB, cc 8.9).
Model: `models/maple-tq2_0.gguf` (5.07 GiB, 20.21 B params, TQ2_0 / 2.06 bpw ternary).
Command: `llama-bench -m models/maple-tq2_0.gguf -ngl 99 -p 512 -n 0 -r 1` (3 runs averaged).

nsys / ncu (NVIDIA Nsight Compute) were unavailable:
- `nsys` not installed.
- `ncu` 2025.1.1 is installed at `C:/Program Files/NVIDIA Corporation/Nsight Compute 2025.1.1/target/windows-desktop-win7-x64/ncu.exe`, but profiling fails with `ERR_NVGPUCTRPERM` (no permission to access GPU performance counters; user is non-admin, so the driver-level GPU counter access required by ncu is denied on Windows WDDM).

Fallback: added cudaEvent-based GPU-time instrumentation directly to the source.

### Instrumentation

`fork/ggml/src/ggml-cuda/ggml-cuda.cu` was patched (only file changed; `convert.cu` and patches untouched) with a `MAPLE_CUDA_PROFILE` block that:

- Records a `cudaEvent_t` before and after each of: src0 dequant, src1 dequant, GEMM, and post-conv (FP16/BF16→FP32) inside `ggml_cuda_op_mul_mat_cublas`. The function dispatches into three branches (BF16, FP16, F32); each branch's events are accumulated into per-path counters.
- After every matmul, `cudaEventSynchronize` is called and `cudaEventElapsedTime` is used to read the GPU-time of each interval; the result is summed into the appropriate counter (`dequant_*_us`, `gemm_*_us`, `post_*_us`, `*_total_us`).
- A static `atexit` handler prints a summary to stderr after llama-bench exits. All code is `#ifdef MAPLE_CUDA_PROFILE` so it's a no-op when the macro is undefined.
- Only `ggml_cuda_op_mul_mat_cublas` is instrumented. `mmq`, `mmvq`, `mmvf`, `mmf` and the per-tensor `mul_mat_*` paths are NOT instrumented. For TQ2_0 + pp512, every weight goes through `mul_mat_cublas` (TQ2_0 is not in the mmq-supported list, batch=512 is too large for mmvq, mmvf rejects quantized types), so this captures essentially 100% of the matmul time.


Across 3 back-to-back runs, t/s varied 870-1025 (GPU thermal/power state) but per-op ratios were stable (dequant 24-26%, gemm 41-45%, post 15-18%, unaccounted ~1%).
The instrumentation adds ~5 µs of CPU-side cudaEventRecord/Synchronize overhead per matmul.
The per-op GPU times measured by `cudaEventElapsedTime` are unaffected by this overhead.

### Why TQ2_0 lands in cublas

In `ggml_cuda_mul_mat` (line ~2610):
- `use_mul_mat_q` is false: `mmq` lists TQ2_0 as `mmq_supported = false` (line 270-308 of `mmq.cu`).
- `use_mul_mat_f` is false: `mmf.cu:135` rejects `ggml_is_quantized` types.
- `use_mul_mat_vec_q` is false: `MMVQ_MAX_BATCH_SIZE = 8`, batch is 512.
- So control falls to `ggml_cuda_op_mul_mat_cublas`.

In `ggml_cuda_op_mul_mat_cublas`, the FP16 fast path is taken (TQ2_0 is quantized, contiguous, row_diff == ne[1], and cc 8.9 has fast FP16). Inside that path, `force_compute_type.fp16` defaults false and the cc-specific override for NVIDIA is empty, so the FP16 *compute* branch (`CUBLAS_COMPUTE_16F`) is used. That branch runs: TQ2_0→FP16 dequant → F32→FP16 src1 cast → cublasGemmEx(FP16, FP16 compute) → FP16→FP32 output conversion.

Empirical cross-check: setting `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` switches the FP16 path to the if-branch (FP16 storage + FP32 compute + FP32 output, no post-conv). Measured: post-conv drops from ~16% to ~7%, but gemm jumps from ~26 µs to ~36 µs/matmul (FP32-compute is ~38% slower than FP16-compute on cc 8.9 for these shapes), and net pp512 t/s is unchanged to slightly better. So the post-conv is **not the right knob to turn** for FP16-quantized weights on cc 8.9.

## Headline numbers

`llama-bench -m models/maple-tq2_0.gguf -ngl 99 -p 512 -n 0 -r 1` × 3 runs.

### Stable across 3 runs

```
calls: total=10963 (fp16=10917 bf16=0 fp32=46)

total matmul time:  ~640 ms (avg ~58 µs/matmul)
dequant src0 (weights):  ~160 ms  (25.0%)
dequant src1 (acts):      ~88 ms  (13.8%)
gemm:                    ~280 ms  (43.5%)   <- cuBLAS FP16 GEMM
post (fp16/bf16->fp32): ~100 ms  (15.7%)
unaccounted:               ~7 ms  ( 1.1%)
```

### FP16 path (TQ2_0 default, 10917 matmuls / 99.6% of calls)

```
Per-matmul average (FP16 path):
  total:                    ~55 µs
  dequant src0 (TQ2_0→FP16): ~14 µs  ( 25.2% )   <-- the 32-thread specialized kernel
  dequant src1 (F32→FP16):    ~8 µs  ( 14.0% )   <-- activation cast (not a real dequant)
  gemm (cublasGemmEx FP16):  ~23 µs  ( 41.0% )   <-- cuBLAS FP16 tensor-core SGEMM
  post (FP16→FP32):           ~9 µs  ( 16.1% )   <-- output conversion
```

### FP32 path (46 matmuls / 0.4% of calls — large/unusual shapes that fail the FP16 fast path)

```
Per-matmul average: ~750 µs, of which the cuBLAS FP32 sgemm is ~99%.
These contribute ~5% of total matmul time despite being <0.5% of call count.
```

## Per-call summary table (averaged across 3 runs)

| Phase                     | Total (ms) | Per-call (µs) | % of matmul |
|---------------------------|-----------:|--------------:|------------:|
| Dequant src0 (TQ2_0→FP16) |     ~161   |       ~15     |       25.2% |
| Dequant src1 (F32→FP16)   |      ~88   |        ~8     |       13.8% |
| cuBLAS GEMM (FP16)        |     ~281   |       ~26     |       43.5% |
| Post-conv (FP16→FP32)     |     ~100   |        ~9     |       15.7% |
| Unaccounted (overhead)    |       ~7   |       ~0.6   |        1.1% |
| **Total matmul**          |    **~637**|     **~58**   |    **100%** |

## Top kernels by time spent (no nsys, so per-op not per-kernel; from the cudaEvent breakdown)

1. **cublasGemmEx FP16** (cuBLAS FP16 tensor-core SGEMM, FP16 inputs / FP16 compute): ~281 ms total, ~26 µs / matmul, **43.5% of matmul time**.
2. **`dequantize_block_tq2_0<half>`** (the 32-thread specialized TQ2_0→FP16 dequant kernel in `convert.cu`, called from `dequantize_row_tq2_0_cuda`): ~161 ms total, ~15 µs / matmul, **25.2%**.
3. **`to_fp32_cuda` (FP16→FP32)** output cast: ~100 ms total, ~9 µs / matmul, **15.7%**.
4. **`to_fp16_cuda` (F32→FP16)** activation cast: ~88 ms total, ~8 µs / matmul, **13.8%**.
5. **cuBLAS setup + kernel launch + cublasSetStream + small overhead**: ~7 ms total, ~0.6 µs / matmul, **1.1%**.

## Where the 5× pp512 gap to q4_k_m comes from

Baseline `pp512` throughputs (profiling disabled, same build directory, run separately):
- TQ2_0: **1398 t/s** (this profile run, with profiling on: 1014-1025 t/s, ~28% slower from sync overhead)
- Q4_K_M: **8130 t/s** (reference, user-supplied); with profiling on this build I measured **8648 t/s** for q4_k_m, but q4_k_m's matmuls go through MMQ, not cublas, so my profile output for q4_k_m is just the 46 cublas calls (the embeddings / output head), which is only ~3% of the model's matmul time. The relevant comparison is unprofiled TQ2_0 vs unprofiled Q4_K_M.

For TQ2_0 + pp512 + RTX 4080 SUPER + 20.21 B params:
- 1398 t/s × 512 tokens = ~715k tokens through the prompt per ~366 ms wallclock budget.
- All matmuls are dispatched through `cublasGemmEx` after dequant, **never through the fast fused `mmq` kernels that Q4_K_M uses**. TQ2_0 has no MMQ implementation, so it pays: read TQ2_0 → write FP16 → read FP16 → GEMM → write FP16 → read FP16 → write FP32. That round-trip through FP16 in HBM is the tax.

If we fuse dequant into the GEMM (the same trick Q4_K_M's MMQ uses), we eliminate the FP16 staging buffer and the FP16→FP32 post-conv. The expected wins:
- Skip `dequant src0` (25%): the MMQ kernel reads TQ2_0 directly.
- Skip `dequant src1` cast (14%) and `post` (16%): the MMQ kernel takes FP32 activations in registers and writes FP32 output directly.
- GEMM time stays roughly the same in µs but the bytes of memory traffic per matmul drop dramatically, which usually also reduces GEMM time because the GEMM is partially memory-bandwidth-bound on these mid-size matmul shapes.

So **the bottleneck is the dequant → FP16-staging → GEMM → FP16→FP32 pipeline**, not a single kernel. The GEMM itself is ~44% of matmul time, but eliminating the dequant/post-conv (which together are ~55%) and keeping a similar GEMM time would more than double throughput.
If we cannot ship an MMQ kernel for TQ2_0 quickly, here is the partial-wins priority order (each line ~independent, after my measured cross-check):

1. **Avoid the FP16 staging buffer by going through the FP32-fallback cublasSgemm** — measured neutral or worse. Forces FP32 GEMM (no tensor cores at this precision), saving 16% post-conv but losing ~38% on the GEMM itself. **No go.**
2. **Force FP32 compute with FP16 storage via `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1`** — measured. Saves the post-conv (16% → 7%) but costs more in the GEMM (FP32-compute is slower than FP16-compute), net pp512 unchanged to slightly better. **No meaningful win.**
3. **Make the TQ2_0 dequant kernel faster** (currently 25%, the user's target of patch 0004): it's already a 32-thread kernel with coalesced FP16 writes. A 64- or 128-thread variant with vectorized FP16 stores, or running on multiple rows per block, could shave 20-40% off the dequant itself = **~5-10% pp512**. Diminishing returns — even a free dequant would only buy 25%.
4. **Fuse dequant + GEMM in a single kernel (MMQ-style for TQ2_0)**: this is what Q4_K_M does and what closes the 5× gap. Expected to reach ~7000-8000 t/s if the MMQ kernel reaches near-cuBLAS SGEMM throughput.

## Recommendation

**Both dequant and GEMM matter, but the actionable bottleneck is the dequant + post-conv pipeline that doesn't exist for q4_k_m.**

For TQ2_0 specifically:
- The dequant kernel itself (`dequantize_block_tq2_0<half>`) is **~25% of matmul time** and worth further tuning as a near-term win. Realistic ceiling: halve it = ~12% pp512.
- The FP16→FP32 post-conv is **~16%** and is *not* part of patch 0004. The cleanest way to kill it is to use `cublasGemmEx` with FP32 output (the if-branch of the FP16 path), but that branch currently isn't taken because `force_compute_type.fp16 = false` and the cc 8.9 NVIDIA override is empty — so the FP16-compute branch (with FP16→FP32 cast) is selected.
- The actual GEMM is **~44%**, comparable to the rest of the pipeline. Optimizing it without fusion is bounded: cuBLAS already uses tensor cores.

**Measured recommendation (above):** setting `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` did not help.

In short — neither "pure dequant" nor "pure GEMM" is the bottleneck in isolation; it is the **dequant → FP16 staging → GEMM → FP16→FP32 post-conv chain** that costs the ~5× vs q4_k_m, because q4_k_m fuses it all into MMQ.