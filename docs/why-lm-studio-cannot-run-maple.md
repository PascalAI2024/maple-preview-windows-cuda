# Why LM Studio cannot run Maple-Preview

This started as a one-line request: *install Maple-Preview in LM Studio*. The
answer turned out to be "you can't," but "you can't" is only useful if you can
show your work. This is the evidence trail, and then the path that does work.

## The model

[`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview) is a
20B-A1B **ternary-weight** Mixture-of-Experts reasoning model (MIT licence).
Ternary means each stored weight is one of `{-1, 0, +1}` scaled by a per-row
factor -- roughly 39% of them are exact zeros. That is what makes a 20B model
fit in 5.45 GB.

| Property | Value |
|---|---|
| Params | 20.2 B total, ~1 B active (A1B) |
| Layers | 24 |
| MoE | 256 experts, top-8, clamp-7 SwiGLU |
| Attention | 3:1 hybrid — SWA-512 : Global Attention |
| RoPE | Partial (64/128) on SWA layers, **none at all** on GA layers |
| Context | 131,072 |

That attention pattern is the crux. A 3:1 sliding-window-to-global ratio where
the global layers carry *no positional encoding whatsoever* is not a
configuration knob on an existing architecture — it is a different compute
graph, and something has to implement it.

## Four artifacts, four dead ends

| Artifact | Format | Verdict |
|---|---|---|
| `deepgrove/maple-preview` | BF16 safetensors + `trust_remote_code` | LM Studio loads only GGUF/MLX — never raw HF checkpoints |
| `deepgrove/maple-preview-2bit-mlx` | MLX | Apple Silicon only; target machine is Windows/CUDA |
| `terasut/maple-preview-GGUF` | GGUF | Repo's own README: *"IN PROGRESS … NOT SUCCESSFUL"* |
| `stamsam/maple-preview-gguf` | GGUF | Works — but **only** under a forked runtime |

The first is worth dwelling on. `config.json` declares:

```json
"architectures": ["MapleForCausalLM"],
"auto_map": {
  "AutoModelForCausalLM": "modeling_maple.MapleForCausalLM"
}
```

`auto_map` means the architecture ships *as Python source in the repo*
(`modeling_maple.py`), executed at load time via `trust_remote_code=True`. LM
Studio has no Python in the loading path at all. There is no version of this
that works.

## The load-bearing claim, verified

The fourth artifact's README says mainline llama.cpp cannot load its files. That
is the single fact the whole question turns on, so it should be checked rather
than believed:

```console
$ curl -sL https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp \
    | grep -i -c "MAPLE"
0
```

Zero. And to prove the grep itself is sound rather than silently matching
nothing:

```console
$ curl -sL .../llama-arch.cpp | grep -oE "LLM_ARCH_(QWEN3MOE|DEEPSEEK2|GRANITE_MOE)" | sort -u
LLM_ARCH_DEEPSEEK2
LLM_ARCH_GRANITE_MOE
LLM_ARCH_QWEN3MOE
```

The grep finds architectures that exist. `maple` is not one of them.

## What LM Studio actually ships

```console
$ lms runtime ls
LLM ENGINE                                        SELECTED    MODEL FORMAT
llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.27.1       ✓            GGUF
llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.24.0                    GGUF
```

A mainline-derived llama.cpp, GGUF only. Loading a Maple GGUF into it produces
`unknown model architecture: 'maple'`.

There are *two* independent blockers, and this matters:

1. **The architecture.** `LLM_ARCH_MAPLE` and its compute graph
   (`src/models/maple.cpp`) exist only in the fork.
2. **The tensor type.** The ternary format is `GGML_TYPE_TQ2_0 = 35`, a ggml
   type mainline has no reader for. Even a hypothetical mainline `maple` arch
   could not decode the weights.

## Could the DLLs just be swapped?

The natural hack — build the fork, drop its DLLs into LM Studio's runtime
folder — does not survive contact with the directory listing:

```
llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.27.1/
├── ggml-base.dll  ggml-cpu.dll  ggml-cuda.dll  llama.dll   <- llama.cpp
├── llm_engine.dll  lmstudiocore.dll  llama-server-impl.dll <- LM Studio's own
└── liblmstudio_bindings_cuda12.node                        <- Node ABI bridge
```

The LM Studio shims are compiled against a *pinned* llama.cpp revision's
internal headers, not the public `llama.h` C API. Introducing a new ggml type
changes type-enum handling across `ggml-base`, `ggml-cpu` and `ggml-cuda`
simultaneously, so all of them would have to move in lockstep with closed-source
binaries that expect the old layout. This is not a supported extension point —
LM Studio has no plugin interface for custom architectures.

**Conclusion: the request is not satisfiable as stated.** Not a configuration
problem, not a missing setting. The runtime does not implement the model.

## What does work

Build the fork and run it as a standalone server. LM Studio stays installed and
untouched; anything that speaks the OpenAI API can point at the new server
instead.

```console
$ grep -n 'LLM_ARCH_MAPLE,' fork/src/llama-arch.cpp
45:    { LLM_ARCH_MAPLE,            "maple"            },

$ grep -n 'TQ2_0' fork/ggml/include/ggml.h
425:        GGML_TYPE_TQ2_0   = 35,
```

Both present. See the [README](../README.md) for the four-script setup.

## Notes for the reader

- Both the model and the fork were published on **2026-08-05**. The fork's own
  status note reads: *"no benchmarks, perplexity, or systematic evals yet."*
  This is day-zero code and should be treated as such.
- The fork is a research port, not a distribution. It publishes **no releases**,
  hence building from source.
- If mainline llama.cpp ever merges a `maple` architecture *and* a `tq2_0`
  reader, LM Studio would pick it up automatically on a runtime update, and this
  entire repo becomes obsolete. That is the good outcome.
