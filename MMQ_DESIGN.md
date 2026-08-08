# TQ2_0 MMQ Kernel Design Report

> Target: a fused dequant + matmul kernel for TQ2_0 that closes the 5x throughput gap versus q4_k_m on RTX 4080 SUPER (sm_89).
> Audience: someone who has never seen llama.cpp internals.
> Status: research only -- no source files were modified.

## 0. Context (why this matters)

Profile on sm_89 (`fork/ggml/src/ggml-cuda/ggml-cuda.cu:102`):

- TQ2_0 pp512 = 1398 t/s
- q4_k_m pp512 = 8130 t/s

The TQ2_0 path (also from `ggml-cuda.cu:102` and the fp16 path comment) is:
- dequant TQ2_0 -> FP16 (25% of total)
- activation cast -> FP16 (14%)
- cuBLAS GEMM in FP16 (44%)
- output post-cast (16%)

q4_k_m fuses everything in ONE MMQ kernel (`fork/ggml/src/ggml-cuda/mmq.cuh`), no FP16 round-trip. Per-matmul budget on sm_89:

- Current TQ2_0 cuBLAS path: ~58 us total
- q4_k_m MMQ equivalent (GEMM-only): ~13 us
- Goal: get TQ2_0 per-matmul from 58 us -> ~25 us (~2x pp512)

The only way to close the gap is an MMQ kernel for TQ2_0.

---

## 1. Block layout recap

Source: `fork/ggml/src/ggml-common.h:282-288`.

```c
// 2.0625 bpw
typedef struct {
    uint8_t qs[QK_K / 4];   // 2 bits per element
    ggml_half d;
} block_tq2_0;
static_assert(sizeof(block_tq2_0) == sizeof(ggml_half) + QK_K / 4, ...);
```

Bytes: 64 bytes of `qs` + 2 bytes fp16 `d` = 66 bytes per block. QK_K = 256 elements per block. ONE scale per block.

### 1.1 Lane decode

The packing is NOT a flat "two bits, low-to-high". Packing comes from `quantize_row_tq2_0_ref` (`fork/ggml/src/ggml-quants.c:2382-2409`):

```c
for (size_t j = 0; j < sizeof(y->qs); j += 32) {
    for (size_t m = 0; m < 32; ++m) {
        uint8_t q = 0;
        for (size_t n = 0; n < 4; ++n) {
            int xi = lroundf(x[m + n*32] * id) + 1;     // -1,0,1 -> 0,1,2
            q += (xi & 3) << (2*n);                    // 2-bit code packed into byte
        }
        y[i].qs[j + m] = q;                            // 4 elements per byte
    }
    x += 4*32;
}
```

Two interleaved strided groups of 32. Element index `e` lives at:

- **byte** = `32*(e/128) + (e%32)` -- i.e. byte 0..31 holds e=0..31 of the first 128, then e=128..159, then back to e=32..63, etc.
- **lane** = `(e/32) % 4` -- which 2-bit field inside the byte

The CUDA comments at `fork/ggml/src/ggml-cuda/convert.cu:148-176` and `fork/ggml/src/ggml-cuda/dequantize.cuh:45-66` capture this exactly:

```c
const int b0 = 32 * (e0 >> 7) + (e0 & 31);  // byte index
const int l0 = (e0 >> 5) & 3;               // 2-bit lane
const int c0 = (x[ib].qs[b0] >> (2 * l0)) & 0x3;
```

### 1.2 Decode formula

`(c - 1) * d` -- same arithmetic as Q2_0. With `c in {0, 1, 2, 3}`, the decoded values are **{-d, 0, +d, +2d}**. The quantizer emits {-d, 0, +d} for ternary weights; +2d only appears for saturated values (see `dequantize.cuh:50-55`).

The decode is a plain per-element subtract (`code - 1`), not a multiply. That matters for the SIMD form: in `vec_dot_tq2_0_q8_1` (`fork/ggml/src/ggml-cuda/vecdotq.cuh:781-820`) it is implemented with `__vsub4(codes, 0x01010101)` so the subtraction does not borrow across the byte boundary (a plain `codes - 1` would corrupt the neighbouring byte).

---

## 2. MMQ infrastructure summary

The MMQ infrastructure is in `fork/ggml/src/ggml-cuda/mmq.cuh` (the kernel template) and `fork/ggml/src/ggml-cuda/mmq.cu` (the dispatch).

### 2.1 What MMQ assumes about a quant type

For each type `T`, MMQ needs:

1. `ggml_cuda_type_traits<T>` with `qk`, `qr`, `qi` (defined at `common.cuh:976-980` for TQ2_0 already).
2. A `VDR_T_Q8_1_MMQ` constant in `vecdotq.cuh` -- "vec dot ratio" = how many contiguous int32 chunks per thread per MMQ step. Q2_0 uses 4, Q4_0 uses 4 (`vecdotq.cuh:113-114`).
3. An entry in `mmq_get_q8_1_ds_layout(T)` (`mmq.cuh:77-119`) -- D4/DS4/D2S6. Picks the Q8_1 staging layout that best fits `T`.
4. An entry in `mmq_get_dp4a_tile_x_sizes(T, mmq_y)` (`mmq.cuh:193-219`) returning `{qs, dm, sc}` ints (32-bit element counts).
5. An entry in `mmq_get_mma_tile_x_k(T)` (`mmq.cuh:235-271`) for the MMA path.
6. `load_tiles_t<mmq_y, need_check>(...)` -- populates shared memory from the raw `x` pointer into the x tile buffer.
7. `vec_dot_t_q8_1_dp4a<mmq_x, mmq_y>(...)` -- does the inner block-level dot product over the K-stripe and accumulates into the per-thread `sum[]`.
8. `mmq_type_traits<...>` specialization for `T` (`mmq.cuh:3443-3451` for Q4_K, `mmq.cuh:3354-3359` for Q2_0) -- picks the right `load_tiles`, `vec_dot_dp4a`, `vec_dot_mma`, and `VDR`.
9. `mmq-instance-t.cu` file with one line: `DECL_MMQ_CASE(GGML_TYPE_T);` (`template-instances/mmq-instance-q4_k.cu:1-4`, `q2_0.cu:1-4`).
10. Switch entries in `ggml_cuda_mul_mat_q_switch_type` (`mmq.cu:6-77`) and `ggml_cuda_should_use_mmq` (`mmq.cu:270-296`).

### 2.2 Tile shapes

From `mmq.cuh`:

- `MMQ_ITER_K = 256` (`mmq.cuh:30`) -- number of K elements loaded per load_tiles call
- `MMQ_TILE_NE_K = 32` (`mmq.cuh:170`) -- the K dimension of the per-iteration stripe
- `MMQ_NWARPS = 8` (`mmq.cuh:32`) -> block size = 8 x warp_size = 256 threads
- `get_mmq_y_host()` returns 128 on Volta+ NVIDIA, 64 elsewhere (`mmq.cuh:143-145`)
- `mmq_x in {8, 16, 24, ..., 128}` chosen at runtime in `mul_mat_q_case` (`mmq.cuh:4170-4217`); `mmq_x_max = 128` when `turing_mma_available(cc)` (sm_75+), else `MMQ_DP4A_MAX_BATCH_SIZE = 64`

### 2.3 Tile shape macros

For each type, the dp4a shared-memory layout is described by a `tile_x_sizes{qs, dm, sc}`. Examples (`mmq.cuh:184-192`):

```c
#define MMQ_DP4A_TXS_Q8_0    {mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K*2/QI8_0 + mmq_y/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q4_K    {mmq_y*MMQ_TILE_NE_K   + mmq_y, mmq_y*MMQ_TILE_NE_K/QI4_K,                    mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}
```

`MMQ_TILE_NE_K + 1` (or `*2 + 1`) is the bank-conflict padding in 32-bit elements: extra column to break the 32-way conflict on shared memory reads.

### 2.4 Kernel structure

Two main entry points (`mmq.cuh`):

- `mul_mat_q<type, mmq_x, need_check>` (`mmq.cuh:3627`): the main kernel. Each block computes one (mmq_x x mmq_y) output tile across one stream-K iteration.
- `mul_mat_q_process_tile<...>` (`mmq.cuh:3536`): the per-tile body. Loads x, loads y, calls vec_dot, accumulates `sum[mmq_x*mmq_y / (nwarps*warp_size)]`, writes back.

The MMA path (`vec_dot_*_mma`) is enabled when `TURING_MMA_AVAILABLE` (sm_75+). For TQ2_0 on sm_89, both paths are available; the dp4a fallback is what Q2_0 uses today (`mmq.cuh:3357`), and that path is the more conservative starting point.

---

## 3. What TQ2_0 already has vs what it needs

### 3.1 Already there (research-only, no edits needed)

| Piece | Location |
|---|---|
| `ggml_cuda_type_traits<GGML_TYPE_TQ2_0>` | `common.cuh:976-980` (qk=QK_K=256, qr=4, qi=8) |
| `dequantize_tq2_0` (per-element fp32 pair) | `dequantize.cuh:45-68` |
| `dequantize_row_tq2_0_cuda` (whole-block FP16) | `convert.cu:147-196` |
| `vec_dot_tq2_0_q8_1` (MMVQ, 32-element chunk) | `vecdotq.cuh:781-820` |
| MMVQ dispatch (`mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_TQ2_0>`) | `mmvq.cu:1030-1034` |
| `VDR_TQ2_0_Q8_1_MMVQ = 1` | `vecdotq.cuh:113` |

### 3.2 What an MMQ needs that TQ2_0 does NOT have

| Piece | Why it's needed |
|---|---|
| `VDR_TQ2_0_Q8_1_MMQ` constant | the dp4a vec_dot is called over many K stripes; VDR says how many chunks per thread per call |
| `load_tiles_tq2_0<mmq_y, need_check>(...)` | reads raw `block_tq2_0[]` from gmem into the x_tile shared buffer in the layout the vec_dot expects |
| `vec_dot_tq2_0_q8_1_dp4a<mmq_x, mmq_y>(...)` | inner block-level dot product that accumulates into `sum[]` across the (mmq_x, mmq_y) tile |
| `mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_TQ2_0>` | trait specialization telling `mul_mat_q_process_tile` which functions to call |
| `mmq-instance-tq2_0.cu` | one-line explicit template instantiation |
| Switch entry in `mmq_get_q8_1_ds_layout` | TQ2_0 has ONE scale per block -> D4 (same as Q2_0) |
| Switch entry in `mmq_get_dp4a_tile_x_sizes` | matches Q2_0 since QK_K=256 with one scale |
| Switch entry in `mmq_get_mma_tile_x_k` | matches Q2_0 (or Q8_0) |
| Switch entry in `ggml_cuda_mul_mat_q_switch_type` | dispatch into the kernel |
| Switch entry in `ggml_cuda_should_use_mmq` | enable MMQ for TQ2_0 |

### 3.3 The vec_dot vs MMQ tile-level difference

`vec_dot_tq2_0_q8_1` is a batch-1 dot product (`vecdotq.cuh:781-820`). It takes ONE 32-element chunk (`iqs in [0,7]`), returns ONE float. MMVQ calls it inside a warp-level reduction (one warp = one output row, `mul_mat_vec_q`).

The MMQ `vec_dot_*_dp4a` does NOT return a float. It computes a (mmq_x, mmq_y) tile-level partial sum, with each thread contributing `mmq_x * mmq_y / (nwarps * warp_size)` floats to a per-thread `sum[]` array. The actual reduction is `mmq_write_back_dp4a` (`mmq.cuh:3444+`). The kernel structure (`mul_mat_q_process_tile`):

```cpp
extern __shared__ int data_mul_mat_q[];
int * tile_y = data_mul_mat_q + mmq_x;
int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);

// outer loop over K iterations
for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, ...);          // x -> shared mem
    for (l0 ...) tile_y[l] = by0[l];                     // y -> shared mem
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, 0);                     // first half of K
    __syncthreads();
    for (l0 ...) tile_y[l] = by0_next[l];                // y (next half)
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);         // second half of K
    __syncthreads();
}
```

So the MMQ `vec_dot` is structurally different from MMVQ `vec_dot`: it reads `x` and `y` from the shared-memory tile buffers (already in the Q8_1 staging layout), iterates over a K-stripe of MMQ_TILE_NE_K=32 elements, and writes into a per-thread accumulator. It cannot just be a wrapper around `vec_dot_tq2_0_q8_1` because:

1. The MMVQ version assumes `y` is a contiguous `block_q8_1[]` in gmem. MMQ has y in `block_q8_1_mmq` (a transposed bank-conflict-free layout with scale/sum slots).
2. MMVQ returns one float per call. MMQ wants to accumulate into `sum[]` across many K iterations.
3. MMVQ processes one chunk at a time (VDR=1 for MMVQ). For MMQ we want a higher VDR (more parallelism across the warp) and a tighter loop over the K stripe.

The MMVQ implementation is still useful as a building block -- specifically the 4-byte SIMD trick (`get_int_b2 -> codes -> __vsub4 -> dp4a`) is exactly what we want in MMQ. See section 7.

---

## 4. Three MMQ approaches, ranked

### Approach A: Adapt the Q4_K MMQ kernel

Q4_K has 4-bit data + per-block mins + 4-bit super-block scales (`block_q4_K`). The MMQ pattern (`mmq.cuh:2179-2313`) handles this with a tile layout that holds qs, dm (half2 scales), and sc (packed sub-block scales). Tile size:

```c
#define MMQ_DP4A_TXS_Q4_K {mmq_y*MMQ_TILE_NE_K + mmq_y, mmq_y*MMQ_TILE_NE_K/QI4_K, mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}
```

**Pros**
- Most general pattern in the codebase; well-tested; supports stream-K; supports MMA path.
- The Q4_K infrastructure already includes everything needed for a complex quant type.

**Cons**
- 90% of the complexity (per-block mins, super-block scale unpack, `unpack_scales_q45_K`) is completely unused by TQ2_0. We would be paying the code size and register cost of `x_dm` (half2 per block), `x_sc` (sub-block scale int per 8 elements), and the `unpack_scales_q45_K` shuffle for a type that has ONE scalar `d` per block.
- The Q4_K `mmq_get_mma_tile_x_k` returns `MMQ_MMA_TILE_X_K_Q8_1` (i.e. it borrows Q8_1's MMA layout). For TQ2_0 we want to reuse Q8_0's layout (since Q2_0 already does this) -- so we would be borrowing a layout we do not actually need.
- The `vec_dot_q4_K_q8_1_impl_mmq` (`vecdotq.cuh:534-557`) does scale/min correction loops that are pure overhead for TQ2_0.

**Register/smem cost** for mmq_y=128 on sm_89:

- Q4_K tile_x: `128*32 + 128 = 4224` ints qs + `128*32/32 = 128` half2 dm + `128*32/8 + 128/8 = 528` ints sc = ~21 KB. With the +1 bank-conflict pad and 8 warps, that is a lot.
- TQ2_0 does not need `dm` or `sc`, only `d` (one fp16 per block -> fits in 128 half2 per row of the tile, which is what Q8_0's layout already provides for d4).

**What would break**
- The Q4_K path assumes `vec_dot_q4_K_q8_1_impl_mmq` is called with `x_dm` and `x_sc` for every K iteration. TQ2_0 has neither. You would be passing garbage and getting NaN -- you would have to stub them out.

**Verdict:** WRONG TOOL. Highest cost, lowest fit.

### Approach B: Adapt the Q2_0 MMQ kernel

Q2_0 is the closest analog: 2-bit per element, one fp16 scale per block, QK=128. The MMQ traits (`mmq.cuh:3354-3359`) reuse `vec_dot_q8_0_q8_1_dp4a` and `load_tiles_q2_0` with `MMQ_DP4A_TXS_Q8_0` (`mmq.cuh:184`). Q2_0's tile size:

```c
#define MMQ_DP4A_TXS_Q8_0 {mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K*2/QI8_0 + mmq_y/(QI8_0/2), 0}
```

That is **2 ints per element** in the qs tile (because Q2_0 expands 2-bit codes to signed bytes via `__byte_perm` so it can use `dp4a` byte-wise).

**Pros**
- Q2_0 already exists; it works on the same hardware; the launch path is identical.
- The shared-memory layout, the MMA shape, and the dp4a path are all already tuned for sm_89.
- `load_tiles_q2_0` (`mmq.cuh:354-415`) has the exact pattern we need: take packed 2-bit codes, expand to signed bytes, store in shared memory in Q8_0 layout.

**Cons**
- **TQ2_0 block is 256 elements, not 128.** Q2_0 has 4 chunks per block; TQ2_0 has 8. MMQ_ITER_K=256 means we load one full TQ2_0 block per iteration (blocks_per_iter = 256/256 = 1) vs Q2_0's 2 blocks per iteration (256/128). That is a register-tile difference that needs adapting `load_tiles`.
- **TQ2_0's lane decode is non-trivial.** Q2_0 packs 4 codes per byte in the natural order: byte 0 holds elements 0..3. TQ2_0 splits the 256 elements across TWO disjoint strided groups of 128 each (`j=0` and `j=32` strides in `quantize_row_tq2_0_ref`), and the lane is `(e/32) % 4` not `e%4`. The `__byte_perm` trick Q2_0 uses (`vecdotq.cuh:741-770`) **does NOT work directly** on TQ2_0's packing -- but `__vsub4(codes, 0x01010101)` followed by `dp4a` does, which is what the MMVQ `vec_dot_tq2_0_q8_1` already does.
- **VQ is identical**: same per-element dot-product arithmetic as Q2_0.

**Register/smem cost** for mmq_y=128, QK_K=256, MMQ_TILE_NE_K=32:

- Tiles fit ONE block per iteration (blocks_per_iter=1), so `load_tiles_tq2_0` loads 8 chunks of 32 bytes (= 256 bytes of qs) plus 1 fp16 d per row.
- qs tile: `128 * (32*2 + 1) = 8448` ints ~ 34 KB. Larger than Q4_K because we go from QK=128 -> QK=256 (2x rows of qs per tile iteration), and each thread still owns its slice.
- Actually TQ2_0 has only `d` (1 fp16 per block), so the `dm` half can use Q2_0/Q8_0's d4 layout: `128 * 32/32 = 128` floats. With the half2 trick that is 64 half2.

**What would break**
- `load_tiles_q2_0` reads 8 bytes (4 int16) per chunk of 32 elements, applies the `__byte_perm(0x020100FF, ..., q >> 0)` table to expand to signed bytes, and writes 8 ints per chunk. For TQ2_0, the 8 bytes per chunk are NOT contiguous in the same way: a "chunk" of 32 elements spans bytes 0..31 (in the first 128) OR bytes 32..63 (in the second 128), and the lane assignment within each byte is `iqs%4`. The MMVQ version reads these as 4 packed bytes via `get_int_b2` and shifts them by `2*lane` bits.
- The Q2_0 `vec_dot_q2_0_q8_1_dp4a` calls the existing `vec_dot_q8_0_q8_1_dp4a`. That is byte-wise dp4a with already-signed bytes. For TQ2_0 we would want the same byte-wise path, but the bytes we hand it are not what Q2_0 produces. We need a TQ2_0-specific load that emits signed bytes per lane.

**Verdict:** BEST STARTING POINT. Right tile size, right arithmetic, right path. ~80% of the code reuses. Need a TQ2_0-specific load_tiles that emits the same signed-byte layout as Q2_0, and a TQ2_0-specific vec_dot wrapper around the same `dp4a`.

### Approach C: Design from scratch using `vec_dot_tq2_0_q8_1` as a building block

Skip the existing Q4_K / Q2_0 structure; write a minimal MMQ tile kernel from the existing `vec_dot_tq2_0_q8_1`.

**Pros**
- Most direct: reuse the existing 4-element-SIMD dp4a form (`get_int_b2 -> codes -> __vsub4 -> dp4a`).
- Can target a smaller register/smem footprint if we know VDR=8 (one chunk = 32 elements = one byte_perm = one dp4a in the MMVQ form, so we just call vec_dot_tq2_0_q8_1 8 times).

**Cons**
- We re-invent the tile loop, the y-staging, the stream-K partitioning, the write-back -- all of which the existing `mul_mat_q_process_tile` already does.
- We would not benefit from the stream-K fixup, the MMA path, the launcher runtime `mmq_x` tuning (`mul_mat_q_case:4170-4217`), the `mmq_get_q8_1_ds_layout` selection, etc.
- The result would be ~2x the LOC of Approach B and would need its own tests/benchmarks; it would not match the upstream merge style.

**Verdict:** STRICTLY WORSE THAN B. Save for "if B fails on register pressure, drop down to C".

---

## 5. Reference paper/algorithm citations (3)

### 5.1 BitNet b1.58 (Ma et al., 2024) -- the algorithm BitNet.cpp implements

Paper: arxiv 2402.17764. Core insight: weights are constrained to {-1, 0, +1}; activations stay at int8; the GEMM is `(A_int8 @ B_ternary^T) * scales`.

What we borrow: the scale-fusion layout -- TQ2_0 single fp16 `d` per 256-element block is the EXACT analog of BitNet b1.58 per-block scale. The MMQ tile loads one block, gets one fp16 scale, multiplies once at the end. Our `vec_dot_tq2_0_q8_1` already implements this pattern: `sumi * (d * d8)`. The MMQ form will accumulate `sumi` over the K stripe and apply the SAME `d * d8` once per tile row.

### 5.2 T-MAC (Wei et al., 2024) -- LUT-based CPU ternary matmul

Paper: arxiv 2407.00088. T-MAC computes ternary matmul on CPUs by building a 256-entry lookup table per output row that maps each 8-bit activation byte to the contribution of the 4 ternary weights at the same position. Then it does 32 table lookups per 32 activations (instead of 32 dp4a-equivalents).

What we borrow: the tile-shape insight -- T-MAC uses an 8-bit activation window x 8 ternary-weight window = 256-entry table, which is the smallest LUT that exploits both bytes' entropy. The dp4a on the GPU does the same thing in hardware: one `dp4a` per 4 (signed weight byte x signed activation byte) sums. Our MMQ design mirrors this: ONE block of 256 TQ2_0 elements x ONE block of 32 Q8_1 elements -> one dp4a chain -> one partial sum, with the per-block fp16 scale applied last.

### 5.3 sparse-ternary-fma / bitnet.cpp -- the practical integration

References:
- llama.cpp Discussion #18821 ("Performance Report: sparse-ternary-fma vs. Original TQ2_0")
- arxiv 2502.11880 ("Bitnet.cpp: Efficient Edge Inference for Ternary LLMs")

What we borrow: the MMVQ-first, MMQ-later progression. TQ2_0 currently uses MMVQ (`vec_dot_tq2_0_q8_1`) for single-token decode and falls through to dequant+cuBLAS for prefill (which is what the 1398 t/s number measures). The bitnet.cpp work confirms the same trade-off: MMVQ is fine for batch=1, but for batch>1 you need fused dequant+GEMM. Adding MMQ is the natural extension of the path that already exists.

---

## 6. Concrete tile shape recommendation

Hardware: RTX 4080 SUPER (sm_89, Ada Lovelace, NVIDIA). The constants we cite are from the code as-is.

| Constant | Value | Source |
|---|---|---|
| MMQ_ITER_K | 256 | mmq.cuh:30 |
| MMQ_TILE_NE_K | 32 | mmq.cuh:170 |
| MMQ_NWARPS | 8 | mmq.cuh:32 |
| get_mmq_y_host() (Volta+) | 128 | mmq.cuh:144 |
| get_mmq_x_max_host() (Turing+) | 128 | mmq.cuh:124-130 |
| mmq_get_granularity_device(128) (Turing, mmq_x>=48) | 16 | mmq.cuh:283-285 |
| qk (TQ2_0) | 256 | common.cuh:977 |
| qi (TQ2_0) | 8 | common.cuh:979 |
| qr (TQ2_0) | 4 | common.cuh:978 |
| blocks_per_iter (TQ2_0) = MMQ_ITER_K/qk = 1 | | derived |
| QK8_1 | 32 | ggml-common.h:253 |
| y blocks per TQ2_0 block = 256/32 = 8 | | derived |

### 6.1 Recommended tile

- M (mmq_y) = 128 -- same as Q4_K and Q2_0 on sm_89.
- N (mmq_x) = 8 -- small because QK_K is already 256 and we only need 1 x-block per iteration. mmq_x=8 means 8 output columns per tile, which gives a healthy number of tiles for an RTX 4080 SUPER's 52 SMs (52 x ceil(8192/8)/2 ~ 26k tiles per layer for a typical 4096-dim hidden -> 8192-dim FFN). The `mul_mat_q_case` runtime tuner (`mmq.cuh:4170-4217`) will pick this or larger.
- K (MMQ_TILE_NE_K = 32) per inner step. The `vec_dot_tq2_0_q8_1_dp4a` inner loop iterates k01 from 0 to MMQ_TILE_NE_K=32 stepping by `QR*VDR`. With QR=4 and VDR=8, each iteration processes 32 elements (= one TQ2_0 chunk), and there is exactly ONE iteration. So the inner K loop is fully unrolled, and we get the same performance as the MMVQ call.
- y tile staging: MMQ_Q8_1_DS_LAYOUT_D4 (same as Q2_0/Q8_0). One scale per 32-element Q8_1 sub-block; no `ds4` sum needed because TQ2_0 has only `d` (no min).

### 6.2 Smem budget

Per block: `mmq_x * MMQ_TILE_Y_K = 8 * 36 = 288` ints for y (MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1 = 32 + 4 = 36, from `mmq.cuh:262`). Plus tile_x: `mmq_y * (2*MMQ_TILE_NE_K + 1) = 128 * 65 = 8320` ints. Total smem ~34 KB per block, well under sm_89's 100 KB shared-memory budget per SM.

### 6.3 Register budget

Each thread holds `mmq_x*mmq_y / (nwarps*warp_size) = 8*128 / (8*32) = 4` floats in `sum[]`. The dp4a inner loop uses 2 ints (`syms` + `q4` from y) and 1 fp16 `d` + 1 fp16 `d8`. Total: ~10 registers of state per thread, well within the 64-register sweet spot for sm_89 (no spills expected).

---

## 7. Block-level dot product algorithm

TQ2_0 K-block: 256 elements packed as 64 bytes (4 codes per byte), 1 fp16 scale.
Q8_1 sub-block: 32 elements packed as 32 int8, 1 half2 scale.

Pseudocode (mirrors `vec_dot_tq2_0_q8_1` at `vecdotq.cuh:781-820` but in the MMQ context -- reads from the x_tile shared buffer that `load_tiles_tq2_0` already populated with signed bytes):

```cpp
// Process ONE K-stripe of MMQ_TILE_NE_K=32 elements.
// Each call to vec_dot accumulates into sum[j0/nwarps*mmq_y/warp_size + i0/warp_size].
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_tq2_0_q8_1_dp4a(
    const int * x, const int * y, float * sum, const int k00) {

    // x is the tile_x shared buffer (signed bytes packed by load_tiles).
    // y is the tile_y shared buffer in block_q8_1_mmq layout.
    // Each k01 step processes 32 elements (one TQ2_0 chunk of 32).
    const half2 * y_ds = (const half2 *) y;                  // 4 half2 per row
    const int   * y_qs = (const int   *) y + 4;              // qs starts after ds
    const int   * x_qs = (const int   *) x + (k00 + k01) / QR_TQ2_0;  // already-signed bytes

    // VDR=8: process 8 chunks of 32 = full 256 elements in one call.
    // Per TQ2_0 chunk, we read 4 packed bytes, extract lane, dp4a with 4 activation bytes.
    int sumi = 0;
    #pragma unroll
    for (int chunk = 0; chunk < VDR_TQ2_0_Q8_1_MMQ; ++chunk) {        // 8 iters
        // load_tiles already shifted codes by 2*lane and stored as int32 per chunk:
        // one int32 holds 4 signed bytes, one byte per element in the chunk.
        const int syms = x_qs[chunk * (MMQ_TILE_NE_K / 4 / 2) + ...];  // 4 signed bytes
        const int acts = y_qs[chunk * (MMQ_TILE_NE_K / 4)];            // 4 activation bytes
        sumi = ggml_cuda_dp4a(syms, acts, sumi);
    }

    // Scale once per row of the tile. d is the TQ2_0 fp16 scale;
    // d8 is the Q8_1 scale; both are pre-staged in shared memory by load_tiles.
    const float d   = __half2float(x_d[i]);
    const float d8  = __half2float(y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
    sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += sumi * (d * d8);
}
```

In actual lines (excluding the load_tiles prep and the per-thread sum indexing), the inner dp4a chain is:

```c
int sumi = 0;
#pragma unroll
for (int chunk = 0; chunk < 8; ++chunk) {       // 8 chunks per 256-elem block
    int syms = x_tile[chunk_idx + chunk];         // 4 signed bytes, one int32
    int acts = y_tile[chunk_idx + chunk];         // 4 int8 activations, one int32
    sumi = ggml_cuda_dp4a(syms, acts, sumi);      // 4x int8 dot product
}
sum[...] += (float)sumi * d * d8;
```

That is **8 dp4a + 1 FMA** per (thread, k-stripe), which is what the MMVQ `vec_dot_tq2_0_q8_1` already does 8x per block, except the MMQ form accumulates across the tile instead of returning.

---

## 8. Phase plan to implementation

Realistic PR-shaped milestone breakdown for an engineer with a CUDA background.

### Phase 0 -- Pre-flight (1 day)

- Confirm sm_89 is the only target for now; document the CC_IS_NVIDIA gating in `mmq.cu`.
- Add a TQ2_0 entry to `ggml_cuda_should_use_mmq` (`mmq.cu:280-296`) with a TODO pointing at the unimplemented kernel; verify the build still compiles (it should -- the trait specialization will just GGML_ABORT at runtime if no `mmq_type_traits<TQ2_0>` exists, which is the current behavior).
- Write a unit test: a 1024x1024 FP32 matmul, quantize to TQ2_0, run both the fp16 path and (after Phase 4) the MMQ path; assert bit-exact agreement to ~1e-3.

### Phase 1 -- Stub the dispatch (1 day)

- Add `mmq_type_traits<...,GGML_TYPE_TQ2_0>` (`mmq.cuh:3443-3451` block) that calls `vec_dot_q8_0_q8_1_dp4a` and `load_tiles_q2_0` -- temporarily reusing Q2_0 functions. This is intentionally broken (wrong tile size, wrong element count) but proves the dispatch wiring.
- Add `mmq-instance-tq2_0.cu` (4 lines).
- Add the three switch entries in `mmq_get_q8_1_ds_layout`, `mmq_get_dp4a_tile_x_sizes`, `mmq_get_mma_tile_x_k` -- use the Q2_0/Q8_0 values.
- Expected outcome: the kernel compiles, launches, produces garbage. Do not merge.

### Phase 2 -- load_tiles_tq2_0 (2 days)

- Implement `load_tiles_tq2_0<mmq_y, need_check>(x, tile_x, kbx0, i_max, stride)`:
  - For each thread, load its slice of `block_tq2_0[i]` qs bytes (64 bytes per block), shift by `2*lane`, store as 4 packed bytes -> 1 int32 of signed bytes into the x_tile shared buffer.
  - Load the per-block `d` (fp16) into the dm slot of the tile.
  - Mirror the Q2_0 pattern (`mmq.cuh:354-415`) but with `get_int_b2` reads of 4 packed bytes per chunk (same as `vec_dot_tq2_0_q8_1`).
- Unit test: write a tiny kernel that loads a single block from a known TQ2_0 buffer and compares the resulting tile_x against an FP16 reference matmul on the same data. Should agree to ~1e-3.

### Phase 3 -- vec_dot_tq2_0_q8_1_dp4a (2 days)

- Implement `vec_dot_tq2_0_q8_1_dp4a<mmq_x, mmq_y>(x, y, sum, k00)` per section 7.
- Inner loop: 8 chunks x 1 dp4a per chunk = 8 dp4a + 1 FMA per (thread, K-stripe).
- Wire it into `mmq_type_traits<TQ2_0>` and remove the Q2_0 stub from Phase 1.
- Unit test: same as Phase 0's test but with this kernel. Bit-exact against the fp16 path.

### Phase 4 -- Bench and tune (2 days)

- Add `VDR_TQ2_0_Q8_1_MMQ = 8` to `vecdotq.cuh`.
- Confirm `mmq_x_best` (chosen at runtime by `mul_mat_q_case:4170-4217`) is in the 8..32 range for sm_89; if not, override the tuner.
- Profile: target >= 6000 t/s at pp512 to match ~2x the current 1398.
- Add the `ggml_cuda_mul_mat_q_switch_type` entry in `mmq.cu:6-77`.
- Revert the experimental `mmvq.cu:254` TQ2_0 short-circuit (the `return 1` that disables MMVQ for TQ2_0 large batches).

### Phase 5 -- MMA path (optional, 3 days)

- sm_89 supports Turing MMA. Adapt `vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>` for TQ2_0 by emitting signed-byte operands. Most of this reuses Q2_0's MMA plumbing.
- This is what q4_k_m uses on sm_89 (Turing+); expected additional speedup is 10-30% over the dp4a path.

### Total effort estimate

- Phase 0-4: 8 working days (~1.5 engineer-weeks) for a working dp4a MMQ with measurable speedup.
- Phase 5: +3 days for the MMA path.
- Testing/benchmarks/docs: +2 days.

---

## 9. Open questions

1. `mul_mat_q_case:4170-4217` runtime tuner. The `mmq_x_best` loop picks the best `mmq_x` for the current problem shape. TQ2_0 may prefer a different range than Q2_0 because QK_K is 2x larger; we will need to extend the candidate set (8..128) or trust the existing loop.
2. `mmvq.cu:254` gating. The current code returns 1 from `mmvq_mmid_max` for TQ2_0, forcing large batches to dequant+cuBLAS. After Phase 4, this should be removed (return the default value) so MMVQ stays available for small batches and MMQ handles large ones.
3. `vec_dot_tq2_0_q8_1` reuse. The MMVQ function still calls `bq8_1[iqs]` directly from gmem, which means the MMVQ path keeps working unchanged. We only add new MMQ-specific code; no edits to existing functions.

---

## 10. Summary

- TQ2_0 already has the per-element MMVQ building block (`vec_dot_tq2_0_q8_1`, `vecdotq.cuh:781-820`).
- An MMQ kernel needs 10 new pieces (section 3.2), of which 5 are one-line switch entries and 2 are small helper functions; the rest is one large function (`load_tiles_tq2_0`) and one medium one (`vec_dot_tq2_0_q8_1_dp4a`).
- The right model to copy is **Q2_0's MMQ** (Approach B), not Q4_K's -- Q2_0 has the same "one fp16 scale, 2-bit codes" structure as TQ2_0.
- For sm_89, tile shape mmq_x=8..32 (chosen at runtime), mmq_y=128, MMQ_TILE_NE_K=32 is the sweet spot. Smem ~34 KB/block; register pressure ~10/thread.
- The block-level dot product (section 7) is 8 dp4a + 1 FMA per (thread, K-stripe), exactly mirroring what MMVQ already does, just accumulated across a tile instead of returned.
- Realistic effort: 8 working days for a working dp4a MMQ with measurable speedup, +3 days for MMA on sm_89.

---

## Appendix A -- File:line index

| Claim | File:line |
|---|---|
| TQ2_0 struct | fork/ggml/src/ggml-common.h:282-288 |
| TQ2_0 type traits | fork/ggml/src/ggml-cuda/common.cuh:976-980 |
| TQ2_0 quantize ref | fork/ggml/src/ggml-quants.c:2382-2409 |
| TQ2_0 dequantize ref | fork/ggml/src/ggml-quants.c:2467-2482 |
| TQ2_0 MMVQ vec_dot | fork/ggml/src/ggml-cuda/vecdotq.cuh:781-820 |
| TQ2_0 dequant CUDA | fork/ggml/src/ggml-cuda/convert.cu:147-196 |
| TQ2_0 dequant per-elem | fork/ggml/src/ggml-cuda/dequantize.cuh:45-68 |
| TQ2_0 MMVQ dispatch | fork/ggml/src/ggml-cuda/mmvq.cu:1030-1034 |
| TQ2_0 fp16 fallback | fork/ggml/src/ggml-cuda/ggml-cuda.cu:102 |
| TQ2_0 MMVQ gating | fork/ggml/src/ggml-cuda/mmvq.cu:252-262 |
| MMQ tile constants | fork/ggml/src/ggml-cuda/mmq.cuh:30, 32, 170 |
| MMQ y staging layout | fork/ggml/src/ggml-cuda/mmq.cuh:60-72 |
| MMQ tile sizes | fork/ggml/src/ggml-cuda/mmq.cuh:184-192 |
| MMQ main kernel | fork/ggml/src/ggml-cuda/mmq.cuh:3627 |
| MMQ process tile | fork/ggml/src/ggml-cuda/mmq.cuh:3536 |
| Q4_K MMQ tile | fork/ggml/src/ggml-cuda/mmq.cuh:2179-2313 |
| Q2_0 MMQ traits | fork/ggml/src/ggml-cuda/mmq.cuh:3354-3359 |
| Q4_K MMQ traits | fork/ggml/src/ggml-cuda/mmq.cuh:3443-3451 |
| Q2_0 load_tiles | fork/ggml/src/ggml-cuda/mmq.cuh:354-415 |
| Q8_0 dp4a vec_dot (reused by Q2_0) | fork/ggml/src/ggml-cuda/mmq.cuh:1204-1225 |
| Q4_K impl_mmq | fork/ggml/src/ggml-cuda/vecdotq.cuh:534-557 |
| Q2_0 MMVQ vec_dot | fork/ggml/src/ggml-cuda/vecdotq.cuh:741-770 |
| MMQ dispatch | fork/ggml/src/ggml-cuda/mmq.cu:6-77 |
| should_use_mmq | fork/ggml/src/ggml-cuda/mmq.cu:270-296 |
| mmq_x tuner | fork/ggml/src/ggml-cuda/mmq.cuh:4170-4217 |
| sm_89 (Ada) | fork/ggml/src/ggml-cuda/common.cuh:55 |
| Turing MMA | fork/ggml/src/ggml-cuda/common.cuh:278-280 |
| q2_0 mmq instance | fork/ggml/src/ggml-cuda/template-instances/mmq-instance-q2_0.cu:1-4 |
| q4_k mmq instance | fork/ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu:1-4 |
| block_q8_1_mmq | fork/ggml/src/ggml-cuda/mmq.cuh:46-72 |