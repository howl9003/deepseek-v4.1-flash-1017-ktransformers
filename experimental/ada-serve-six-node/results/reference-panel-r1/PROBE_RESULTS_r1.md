# DeepSeek's reference kernels on the 4080 SUPER, 1 Nov 2026

DeepSeek's `inference/kernel.py`, unmodified (sha256 1236c350, same as the Hub), compiles and runs on Ada. The only failure is the one GB10 also needed a shim for. This is the first time the reference kernels have run here. No model, no weights, about a minute of card time.

Tilelang 0.1.12, apache-tvm-ffi 0.1.11, torch 2.14.1 cu130, CUDA 13.3, `sm_89`. Each case runs in its own process, 16 repeats. Synthetic inputs. Script `probe_ref_kernels_r3.py`; evidence on fea01 in `/scratch/aster/bench/reference-panel-r1/`.

| Kernel | Runs | Repeats equal | Against an exact check |
|---|---|---|---|
| `act_quant`, FP8 output | Yes | Yes | 0 of 343,040 values, 0 scales |
| `act_quant`, in place | Yes | Yes | 0 of 343,040 |
| `fp4_act_quant`, E8M0 per 32 | Yes | Yes | 0 of 68,608 |
| `fp4_act_quant`, E4M3 per 16 | Yes | Yes | 0 of 68,608 |
| `fp4_act_quant`, packed | Yes | Yes | Dtypes and shape only |
| `fp8_gemm`, M 67, N 256, K 5120 | Yes | Yes | 93.63% of BF16 outputs equal exact math rounded |
| `fp4_gemm`, M 67, N 512, K 5120 | Yes | Yes | 98.26% |
| `sparse_attn`, 64 heads | **No** | | Needs 141,312 B of shared memory; the card refused |
| `sparse_attn`, 16 heads | Yes | Yes | Relative L2 2.2e-3 |
| `hc_split_sinkhorn` | Yes | Yes | Relative L2 9.8e-8 |

## Reading it

- **The reference panel can run on this card.** `sparse_attn` needs the same shim sudoingX uses on GB10: call it per 16-head group. Heads are independent, so that split is exact by their argument. I have not checked it against a 64-head run, which this card cannot do.
- **FP4 works on Ada.** I had expected the casts might not compile, since our fused FP4 kernels need `sm_100+`. Tilelang's casts go through a path that compiles here. The quantisers are exact, including E2M1 ties (0.75 goes to 1.0, the even code).
- **No race seen.** Other cards patch `act_quant` to `num_stages = 2`. The unpatched kernel gave equal bytes on all 16 repeats here. That is not proof the race cannot happen on Ada, only that this test did not show it.
- **No row-count dependence in the GEMMs.** Rows came out with the same bits at M = 1, 2, 16, 32, 33, 64 against M = 67. So Baekpica's batching noise is not coming from these two kernels on this card. The whole forward does show it, though: see `REFERENCE_PIECES_RESULTS_r1.md`. Changing the MoE batch moves 94% of positions, so the sensitivity is elsewhere in the forward.
- **The GEMM gaps are Ada's truncation plus FP32 rounding.** 6.4% of dense and 1.7% of expert outputs round to a different BF16 value from exact math, on these synthetic inputs. Relative L2 1.7e-3 is the BF16 rounding itself. The share depends on the inputs' spread, so it is not a figure for real activations.

## Oracle fixes on the way

r1 and r2 showed mismatches that were my checks, not the kernels. r1 marked 0.5 as the even E2M1 code; it is code 1. r1 and r2 used `torch.log2` for the exponent, which is not exact on powers of two, so groups with amax 7 x 2^k or 6 x 2^k landed one octave high. r3 uses `frexp` and shows none. Our contract uses the same bit-inspection ceil as the reference (`contract_a.py`).

## What it does not show

The bits here are DeepSeek's code on Ada, not on Hopper or Blackwell, so a panel reference built on this card carries Ada's truncation. Runtime on real shapes is unknown. Nothing here runs the whole forward.

## Next, for the panel run

A `Checkpoint` adapter reading our GGUF experts, `original-dense-*` and `engram-v41-fp8` (verify those are the 1017 tables), the per-16-head shim, then `p4_kld.py --mode ref`. Then a capture overlay on our server.
