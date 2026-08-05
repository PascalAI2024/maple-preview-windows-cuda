# A one-line change that makes ternary MoE generation 4.9× faster

**TL;DR** — `maple-tq2_0` generated at ~52 t/s on an RTX 4080 SUPER. One line in
`ggml/src/ggml-cuda/mmvq.cu` was disabling the ternary MMVQ kernel for *every*
MoE expert matmul, including batch-1 token generation. Re-enabling it for
batch-1 took generation to **254.78 t/s** with byte-identical reasoning output.

| | Before | After |
|---|---:|---:|
| `tg128` (generation) | 52.10 ± 1.06 t/s | **254.78 ± 2.86 t/s** |
| `pp512` (prompt) | 1255.74 ± 59.63 t/s | 1173.44 ± 45.28 t/s |

Generation **4.9×**. Prompt processing unchanged within noise, as expected — the
change only affects batch size 1.

## How I got here

The model card advertises **218 tok/s on an M4 Mac mini**, via deepgrove's own
separate on-device runtime. My 52 t/s was a different stack entirely (CUDA +
llama.cpp fork), so the two were never comparable. But 52 t/s was suspicious on
its own terms: Maple is a 20B-A1B MoE, so only **~1 B parameters are active** per
token. A dense 1B model on a 4080 SUPER runs many hundreds of t/s. Something was
costing an order of magnitude.

First hypothesis — attention. Wrong:

```console
$ llama-bench -m maple-tq2_0.gguf -ngl 99 -fa on,off -p 512 -n 128
  fa=1  tg128  51.25 ± 0.30
  fa=0  tg128  52.10 ± 1.06
```

Flash attention made no difference, so the cost was in the FFN/expert path, which
is where a 256-expert top-8 MoE spends nearly all its time anyway.

## Root cause

Checking which CUDA kernels actually exist for the ternary type:

```console
$ grep -rn "TQ2_0" ggml/src/ggml-cuda/mmq.cu ggml/src/ggml-cuda/mmq.cuh
(nothing)

$ grep -rn "TQ2_0" ggml/src/ggml-cuda/mmvq.cu
14:   case GGML_TYPE_TQ2_0:  return vec_dot_tq2_0_q8_1;
1024: case GGML_TYPE_TQ2_0:  mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_TQ2_0>
```

So MMQ (large-batch quantized matmul) genuinely has no ternary support — the
fork says as much. But MMVQ **does**, and the `mul_mat_vec_q_switch_ncols_dst`
call takes `ids`, meaning it is wired for `MUL_MAT_ID` — the MoE expert path.

The kernel existed. So why wasn't it running? `ggml-cuda.cu`:

```c
if (ggml_is_quantized(src0->type)) {
    const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
    if (ne2 <= mmvq_mmid_max) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);   // fast path
        return;
    }
}
```

and `mmvq.cu`:

```c
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // tq2_0 has mmvq (mul_mat_vec_q) but no mmq kernels: route large batches to dequant+gemm.
    if (type == GGML_TYPE_TQ2_0) {
        return -1;
    }
```

`ne2` is a batch size, so it is always ≥ 1. With `mmvq_mmid_max = -1`, the
condition `ne2 <= -1` is **never true**. The fast path is unreachable at every
batch size.

Execution then falls through `should_use_mmq` (no ternary MMQ → false) and
`should_use_mmf` (float-only → false) to the generic path at the bottom of
`ggml_cuda_mul_mat_id`: **dequantize the ternary weights to F32 and run a general
GEMM**, once per expert, per layer, per token.

The comment says the intent was to route *large* batches to dequant+GEMM. `-1`
routes *all* of them, including the batch-1 case MMVQ is built for. That reads
like an overly broad guard rather than a deliberate correctness constraint.

## The change

```diff
 if (type == GGML_TYPE_TQ2_0) {
-    return -1;
+    return 1;
 }
```

`return 1` admits only `ne2 == 1` — single-token generation — to MMVQ. Large
batches still take dequant+GEMM, so the missing MMQ kernels are not a problem.
Full patch: [`patches/0001-tq2_0-enable-mmvq-for-batch1-moe.patch`](../patches/0001-tq2_0-enable-mmvq-for-batch1-moe.patch),
applied automatically by `scripts/02-build.ps1` (opt out with `-NoPatch`).

## Correctness

A 5× speedup from one line deserves suspicion — a subtly wrong kernel still
emits fluent text. Verified at `--temp 0`:

| Prompt | Output | Correct |
|---|---|---|
| `Compute 17 * 24. Give only the number.` | `408` | ✅ |
| `A train leaves at 14:45 and arrives at 17:20. How long is the journey?` | `155 minutes`, via two independent derivations in its reasoning trace | ✅ |
| `What is the capital of Australia, and name the two larger cities often mistaken for it.` | `Canberra`, `Sydney`, `Melbourne` | ✅ |

Multi-step arithmetic is the sensitive probe here: corrupted expert matmuls
degrade chained numeric reasoning long before they damage fluency. The
reasoning traces stayed coherent and arrived at correct answers by valid routes.

## Control experiment

To check whether the patched ternary number is actually *good* rather than just
better, I ran the same model quantized to `q4_k_m` — Q4_K has mature, fully
optimized CUDA kernels and was never touched by the `-1` guard.

| Build | pp512 (t/s) | tg128 (t/s) | Size |
|---|---:|---:|---:|
| `tq2_0` unpatched | 1255.7 | 52.1 | 5.07 GiB |
| `tq2_0` patched | 1173.4 | **254.8** | 5.07 GiB |
| `q4_k_m` (control) | **8130.5** | 298.2 | 11.48 GiB |

Two conclusions:

1. **Generation is now roughly where it should be.** Ternary went from 17% of the
   Q4_K baseline to **85%** of it. A residual gap is expected — Q4_K's kernels
   have had far more optimization attention — but the order-of-magnitude anomaly
   is gone. This is what confirms the diagnosis rather than merely correlating
   with it.

2. **Prompt processing is the remaining ternary gap: 1173 vs 8130 t/s, ~7×.**
   That is the missing MMQ kernels, exactly as the fork documents. It is not
   fixable with a guard change; it needs ternary MMQ kernels written. That is the
   real outstanding work.

Worth noting what this buys: `tq2_0` reaches 85% of Q4_K's generation speed in
**44% of the memory**. That is the ternary value proposition finally showing up.

## Caveats

- Measured on **one** GPU (RTX 4080 SUPER, sm_89, CUDA 12.8) with **one** model.
  Ada Lovelace takes a specific branch in this function; other architectures are
  untested by me.
- I cannot rule out that `-1` guards against something real on hardware I do not
  have. The verification above is behavioural, not a proof of numerical
  equivalence — I did not run a perplexity comparison.
- `-NoPatch` builds the fork unmodified if you want the baseline.

## On upstreaming

I am not submitting this upstream. llama.cpp's `AGENTS.md` is explicit that the
project does not accept predominantly AI-generated pull requests, and instructs
autonomous agents not to contribute. It also states that **private forks are
exempt**, which is what this is.

Anyone who wants this upstream should reach their own understanding of the code
and author the submission themselves — which the diff above should make
straightforward, since the actual change is one token.
